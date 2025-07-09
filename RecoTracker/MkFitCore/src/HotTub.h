#ifndef RecoTracker_MkFitCore_src_HotTub_h
#define RecoTracker_MkFitCore_src_HotTub_h

#include <algorithm>
#include <exception>
#include <vector>

// A smallish pool of objects, reusing the most recently used ones first (for,
// presumably, improved cache reuse). The underlying memory is stored in
// std::vector<> of the templated type. This means whole memory gets re-allocated
// when the HotTub needs to grow, invalidating references and pointers to its content.
// This is intentional as doing it in chunked alloation might result in chunks
// being allocated in different threads and with different VM mappings which would
// mean that items in the hot-tub can not serve as input for vgather-like instructions.

namespace mkfit {

    /**
     * HutTubItem -- base class for object stored in a HotTub
     */
    struct HotTubItem {
      // When "in-use", these are used by the HotTubConsumer to manage its object list.
      // m_ht_slot is set by HotTub, the other two are set by HotTubConsumer.
      // When the item/slot is released, m_ht_next_slot is used to form a stack of
      // free items. m_ht_slot is set to -1 to mark a "not-in-use" item.
      int m_ht_slot = -1;
      int m_ht_prev_slot = -1;
      int m_ht_next_slot = -1;

      // block overriding of internal members.
      HotTubItem& operator=(const HotTubItem &o) { return *this; }
    };

    /**
     * HotTub template -- small pool of objects
     */
    template<class T>
    class HotTub {
    public:
      HotTub(int initial_capacity = s_alloc_atom) {
        reserve(std::max(s_alloc_atom, initial_capacity));
      }

      T& operator[](int i) { return m_vec[i]; }
      const T& operator[](int i) const { return m_vec[i]; }

      void clear() {
        m_size = 0;
        m_peak_slot = 0;
        m_free_slot = -1;
      }

      void reserve(int capacity) {
        // round up capacity to nearest alloc atom
        int rem = capacity % s_alloc_atom;
        if (rem) {
          capacity += s_alloc_atom - rem;
        }
        if (m_capacity < capacity) {
          if (m_throw_on_realloc) {
            throw std::runtime_error("reallocation in critical section");
          }
          m_vec.resize(capacity);
          m_capacity = capacity;
        }
      }

      // To avoid reallocation in critical sections, where ptrs or refs are held
      // to pool items.

      void reserve_for_extra(int n_items) {
        reserve(m_size + n_items);
      }

      void throw_on_realloc(bool flag) {
        m_throw_on_realloc = flag;
      }

      // get / release an item slot.

      int get_slot() {
        int slot;
        if (m_free_slot >= 0) {
          slot = m_free_slot;
          m_free_slot = m_vec[slot].m_ht_next_slot;
        } else {
          if (m_peak_slot >= m_capacity) {
            reserve(2 * m_capacity);
          }
          slot = m_peak_slot++;
        }
        ++m_size;
        m_vec[slot].m_ht_slot = slot;
        return slot;
      }

      void release_slot(int slot) {
        --m_size;
        m_vec[slot].m_ht_slot = -1;
        m_vec[slot].m_ht_next_slot = m_free_slot;
        m_free_slot = slot;
      }

      T& get_item() {
        return m_vec[ get_slot() ];
      }

      void release_item(T& item) {
        release_slot(item.m_ht_slot);
      }

    protected:
      std::vector<T> m_vec;
      int m_capacity = 0;    // How many slots we can address, size of the vector.
      int m_size = 0;        // Number of slots in use, can be non-consecutive.
      int m_peak_slot = 0;   // First free slot at the final extent of free slots.
                             // Only used for inital growth, afterwards free-slot-list is used.
      int m_free_slot = -1;  //
      bool m_throw_on_realloc = false;

    private:
      static constexpr int s_alloc_atom = 4;
    };

    /**
     * HotTubConsumer
     * Uses objects from the HotTub as double-linked-list.
     * Could be vector<int>, too -- or it could even use both at the same time.
     * Hmmh, or it could be in different classes, like HotTubConsumerList and Vector.
     * Knows the number of used objects.
     * One should use access through index when new items are being added into the tub
     * and it is possible that the tub might need to reallocate.
     */
    template<class T>
    class HotTubConsumer {
    public:
      HotTubConsumer(HotTub<T>& the_tub) : m_tub(the_tub) {}

      // Get an independent item -- it still needs to be inserted into the list (or released
      // manually).
      T& get_item() {
        return m_tub.get_item();
      }
      void release_item(T &item) {
        m_tub.release_item(item);
      }

      void push_front(T &item) {
        item.m_ht_next_slot = m_first_slot;
        m_first_slot = item.m_ht_slot;
        item.m_ht_prev_slot = -1;
        if (m_size == 0)
          m_last_slot = m_first_slot;
        ++m_size;
      }
      void push_back(T &item) {
        item.m_ht_prev_slot = m_last_slot;
        m_last_slot = item.m_ht_slot;
        item.m_ht_next_slot = -1;
        if (m_size == 0)
          m_first_slot = m_last_slot;
        ++m_size;
      }
      void insert_before(T &refi, T &item) {
        item.m_ht_prev_slot = refi.m_ht_prev_slot;
        item.m_ht_next_slot = refi.m_ht_slot;
        refi.m_ht_prev_slot = item.m_ht_slot;
        if (m_first_slot == refi.m_ht_slot)
          m_first_slot = item.m_ht_slot;
        ++m_size;
      }
      void insert_after(T &refi, T &item) {
        item.m_ht_prev_slot = refi.m_ht_slot;
        item.m_ht_next_slot = refi.m_ht_next_slot;
        refi.m_ht_next_slot = item.m_ht_slot;
        if (m_last_slot == refi.m_ht_slot)
          m_last_slot = item.m_ht_slot;
        ++m_size;
      }

      void pop_front() {
        assert(m_first_slot >= 0);
        T &item = m_tub[ m_first_slot ];
        if (item.m_ht_next_slot >= 0) {
          T &next = m_tub[ item.m_ht_next_slot ];
          next.m_ht_prev_slot = - 1;
          m_first_slot = next.m_ht_slot;
        } else {
          m_first_slot = m_last_slot = -1;
        }
        --m_size;
        release_item(item);
      }
      void pop_back() {
        assert(m_last_slot >= 0);
        T &item = m_tub[ m_last_slot ];
        if (item.m_ht_prev_slot >= 0) {
          T &prev = m_tub[ item.m_ht_prev_slot ];
          prev.m_ht_next_slot = - 1;
          m_last_slot = prev.m_ht_slot;
        } else {
          m_first_slot = m_last_slot = -1;
        }
        --m_size;
        release_item(item);
      }
      void remove(T &item) {
        assert(m_size > 0);
        if (item.m_ht_prev_slot < 0) {
          return pop_front();
          return;
        }
        if (item.m_ht_next_slot < 0) {
          return pop_back();
          // return;
        }
        T &prev = m_tub[ item.m_ht_prev_slot ];
        T &next = m_tub[ item.m_ht_next_slot ];
        prev.m_ht_next_slot = item.m_ht_next_slot;
        next.m_ht_prev_slot = item.m_ht_prev_slot;
        --m_size;
        release_item(item);
      }

      int size() const { return m_size; }
      bool empty() const { return m_size == 0; }

    private:
      HotTub<T> &m_tub;
      int m_first_slot = -1;
      int m_last_slot = -1;
      int m_size = 0;
    };

} // namespace mkfit

#endif

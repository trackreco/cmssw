#include "RdfSources.h"

#include <TClass.h>
#include <TDataMember.h>

#include <typeinfo>
#include <stdexcept>

namespace mkfit {

  // Helper template to extract type info from TClass members
  template <typename T>
  std::vector<std::string> GetColumnNamesFromClass() {
    std::vector<std::string> names;
    TClass *cls = TClass::GetClass(typeid(T));
    if (!cls) return names;

    TIter next(cls->GetListOfDataMembers());
    TDataMember *member;
    while ((member = dynamic_cast<TDataMember *>(next()))) {
      if (!member->IsaPointer()) {
        names.push_back(member->GetName());
      }
    }
    return names;
  }

  // Helper template to get type name for a specific column
  template <typename T>
  std::string GetTypeNameForColumn(std::string_view columnName) {
    TClass *cls = TClass::GetClass(typeid(T));
    if (!cls) return "";
    TDataMember *member = cls->GetDataMember(columnName.data());
    if (!member) return "";
    return member->GetTypeName();
  }

  class LambdaColumnReader : public ROOT::Detail::RDF::RColumnReaderBase {
    private:
    std::function<void* ()> lambda_;

    public:
    explicit LambdaColumnReader(std::function<void* ()> lambda) : lambda_(std::move(lambda)) {}

    void *GetImpl(Long64_t gimpl_entry) override {
      return lambda_();
    }
  };

  // ================================================================
  #pragma region VectorBackedRDataSource Template Implementation
  // ================================================================

  template <typename T>
  class VectorBackedRDataSource final : public ROOT::RDF::RDataSource {
    public:
    using Vec_t = std::vector<T>;

    explicit VectorBackedRDataSource(const Vec_t* vec) : fVec(vec) {}

    std::string GetLabel() override {
        return std::string("VectorBackedRDataSource<") + std::string(typeid(T).name()) + ">";
    }

    // --------------------------------------------------

    std::vector<std::pair<ULong64_t, ULong64_t>>
    GetEntryRanges() override {
      if (!fVec || fNextEntry >= fVec->size()) {
        if (fDebugLevel > 0) {
          printf("VectorBackedRDataSource::GetEntryRanges %s (fNextEntry = %llu, total entries = %zu)\n",
                 fVec ? "end of vector reached" : "vector is null",
                 fNextEntry, fVec ? fVec->size() : 0);
        }
        return {};
      }
      fNextEntry = fVec->size();
      if (fDebugLevel > 0) {
        printf("VectorBackedRDataSource::GetEntryRanges returning range [0, %zu]\n", fVec->size());
      }
      return {{0, fVec->size()}};
    }

    void SetNSlots(unsigned int nSlots) override {
      if (fDebugLevel > 1) {
        printf("VectorBackedRDataSource::SetNSlots called with nSlots = %u\n", nSlots);
      }
      fSlotEntry.resize(nSlots, 0);
    }

    void Initialize() override {
      if (fDebugLevel > 0) {
        printf("VectorBackedRDataSource::Initialize called for %s\n", GetLabel().c_str());
      }
      fNextEntry = 0;
    }
    void InitSlot(unsigned int slot, ULong64_t firstEntry) override {
      if (fDebugLevel > 1) {
        printf("VectorBackedRDataSource::InitSlot called for slot %u, firstEntry %llu\n", slot, firstEntry);
      }
    }
    bool SetEntry(unsigned int slot, ULong64_t entry) override {
      if (fDebugLevel > 2) {
        printf("VectorBackedRDataSource::SetEntry called for slot %u with entry %llu\n", slot, entry);
      }
      fSlotEntry[slot] = entry;
      return true;
    }
    void FinalizeSlot(unsigned int slot) override {
      if (fDebugLevel > 1) {
        printf("VectorBackedRDataSource::FinalizeSlot called for slot %u\n", slot);
      }
    }
    void Finalize() override {
      if (fDebugLevel > 0) {
        printf("VectorBackedRDataSource::Finalize called for %s\n", GetLabel().c_str());
      }
    }

    // --------------------------------------------------
    // Column Introspection via Dictionary
    // --------------------------------------------------

    const std::vector<std::string>& GetColumnNames() const override {
      static std::vector<std::string> names = GetColumnNamesFromClass<T>();
      return names;
    }

    std::string GetTypeName(std::string_view col) const override {
      return GetTypeNameForColumn<T>(col);
    }

    bool HasColumn(std::string_view col) const override {
      auto cls = TClass::GetClass(typeid(T));
      return cls && cls->GetDataMember(col.data());
    }

    // --------------------------------------------------
    // Column Reader
    // --------------------------------------------------

    std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
    GetColumnReaders(unsigned int slot, std::string_view columnName, const std::type_info&) override
    {
      auto cls = TClass::GetClass(typeid(T));
      auto member = cls->GetDataMember(columnName.data());
      const int offset = member->GetOffset();

      auto lambda = [this, slot, offset]() -> void* {
        const auto entry = fSlotEntry[slot];
        if (entry >= fVec->size()) return nullptr;

        return (void*)(
          reinterpret_cast<char*>(
            const_cast<T*>(&(*fVec)[entry])
          ) + offset
        );
      };

      return std::make_unique<LambdaColumnReader>(lambda);
    }

    // Old API - not used but abstract in base
    std::vector<void*> GetColumnReadersImpl(std::string_view, const std::type_info&) override { return {}; }

    private:
    const Vec_t* fVec = nullptr;
    ULong64_t fNextEntry = 0;
    std::vector<ULong64_t> fSlotEntry;
    int fDebugLevel = 1; // 0 = no debug, 1 = basic, 2 = full, 3 = including SetEntry calls
  };

  #pragma endregion
  // ================================================================
  #pragma region EventSource
  // ================================================================

  /**
   * @brief A custom RDataSource that wraps a single Event or a collection of Events.
   *
   * Design: One RDF entry = one mkfit::Event.
   * The data source exposes a single column "event" of type const mkfit::Event*.
   * All other data (seeds, hits, etc.) is accessed via lambda projections on this pointer.
   */
  class EventSource : public ROOT::RDF::RDataSource {
  public:
    explicit EventSource(std::vector<const Event*>& events) : fEvents(events) {
      if (!fEvents.empty()) {
        fNextEntry = 0;
      }
    }

    ~EventSource() override = default;

    std::string GetLabel() override {
      return "EventSource";
    }

    // --------------------------------------------------

    std::vector<std::pair<ULong64_t, ULong64_t>>
    GetEntryRanges() override {
      ULong64_t ev_size = fEvents.size();

      if (fEvents.empty() || fNextEntry >= ev_size) {
        if (fDebugLevel > 0) {
          printf("EventSource::GetEntryRanges called ... vector is null or fNextEntry out of bounds\n");
        }
        return {};
      }
      std::vector<std::pair<ULong64_t, ULong64_t>> out;
      ULong64_t ns = fSlotEntry.size();
      ULong64_t nentries = ev_size - fNextEntry;
      ULong64_t atom = std::max(nentries / ns, 1ull);
      while (fNextEntry < ev_size) {
        ULong64_t beg = fNextEntry;
        ULong64_t end = std::min(beg + atom, ev_size);
        if (fDebugLevel > 0) {
          printf("EventSource::GetEntryRanges appending range [%llu, %llu]\n", beg, end);
        }
        out.push_back( { beg, end } );
        fNextEntry = end;
      }
      return out;
    }

    void SetNSlots(unsigned int nSlots) override {
      if (fDebugLevel > 1) {
        printf("EventSource::SetNSlots called with nSlots = %u\n", nSlots);
      }
      fSlotEntry.resize(nSlots, nullptr);
    }

    void Initialize() override {
      fNextEntry = 0;
      if (fDebugLevel > 0) {
        printf("EventSource::Initialize called for %s\n", GetLabel().c_str());
      }
    }

    void InitSlot(unsigned int slot, ULong64_t firstEntry) override {
      if (fDebugLevel > 1) {
        printf("EventSource::InitSlot called for slot %u, firstEntry %llu\n", slot, firstEntry);
      }
    }

    bool SetEntry(unsigned int slot, ULong64_t entry) override {
      if (fDebugLevel > 2) {
        printf("EventSource::SetEntry called for slot %u with entry %llu\n", slot, entry);
      }
      if (entry >= fEvents.size()) return false;
      fSlotEntry[slot] = fEvents[entry];
      return true;
    }

    void FinalizeSlot(unsigned int slot) override {
      if (fDebugLevel > 1) {
          printf("EventSource::FinalizeSlot called for slot %u\n", slot);
      }
    }

    void Finalize() override {if (fDebugLevel > 0) {
        printf("EventSource::Finalize called for %s\n", GetLabel().c_str());
      }
    }

    // --------------------------------------------------
    // Column Introspection -- trivial
    // --------------------------------------------------

    const std::vector<std::string>& GetColumnNames() const override {
      static std::vector<std::string> names = { "event" };
      return names;
    }

    std::string GetTypeName(std::string_view col) const override {
      return "const mkfit::Event*";
    }

    bool HasColumn(std::string_view col) const override {
      return (col == "event");
    }

    // --------------------------------------------------
    // Column Reader
    // --------------------------------------------------

    std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
    GetColumnReaders(unsigned int slot, std::string_view columnName, const std::type_info&) override {
      if (columnName != "event")
        return nullptr;

      auto lambda = [this, slot]() -> void* {
        return (void*) &fSlotEntry[slot];
      };

      return std::make_unique<LambdaColumnReader>(lambda);
    }

    // Old API - not used but abstract in base
    std::vector<void*> GetColumnReadersImpl(std::string_view, const std::type_info&) override { return {}; }

  private:
    std::vector<const Event*> fEvents;
    std::vector<const Event*> fSlotEntry;
    ULong64_t fNextEntry = 0;
    int fDebugLevel = 1; // 0 = no debug, 1 = basic, 2 = full, 3 = including SetEntry calls
  };

  #pragma endregion
  // ================================================================
  #pragma region RdfSources
  // ================================================================

  ROOT::RDataFrame RdfSources::MakeTrCandMetaDF(const Event &ev) {
    return ROOT::RDataFrame(std::make_unique<VectorBackedRDataSource<TrCandMeta>>(&ev.trCandMetas_));
  }
  ROOT::RDataFrame RdfSources::MakeTrCandStateDF(const Event &ev) {
    return ROOT::RDataFrame(std::make_unique<VectorBackedRDataSource<TrCandState>>(&ev.trCandStates_));
  }
  ROOT::RDataFrame RdfSources::MakeTrHitMatchDF(const Event &ev) {
    return ROOT::RDataFrame(std::make_unique<VectorBackedRDataSource<TrHitMatch>>(&ev.trHitMatches_));
  }

  ROOT::RDataFrame RdfSources::MakeTrackDF(const TrackVec &tvec) {
    return ROOT::RDataFrame(std::make_unique<VectorBackedRDataSource<Track>>(&tvec));
  }
  ROOT::RDataFrame RdfSources::MakeSeedDF(const Event &ev) {
    return MakeTrackDF(*ev.currentSeedTracks_);
  }

  ROOT::RDataFrame RdfSources::MakeEventDF(std::vector<const Event*>& events) {
    if (events.empty()) return ROOT::RDataFrame(0);
    return ROOT::RDataFrame(std::make_unique<EventSource>(events));
  }

  #pragma endregion
}  // end namespace mkfit

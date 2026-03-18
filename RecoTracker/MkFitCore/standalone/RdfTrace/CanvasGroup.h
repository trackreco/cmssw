#ifndef RecoTracker_MkFitCore_standalone_RdfTrace_CanvasGroup_h
#define RecoTracker_MkFitCore_standalone_RdfTrace_CanvasGroup_h

#include "RecoTracker/MkFitCore/standalone/Event.h"

#include "ROOT/RDataFrame.hxx"

class TCanvas;
class TVirtualPad;

extern TCanvas *canvas_ptr;
extern int canvas_i, canvas_imax;

void np();
void nc(int dx=1, int dy=1, const char *n=0, const char *t=0, const char *pfx=0);

// CanvasCroup -- to hold a group of histograms and draw them together
struct CanvasGroup
{
  using predraw_mod_func = std::function<void(TH1*, TVirtualPad*)>;
  using postdraw_mod_func = std::function<void(TH1*, TVirtualPad*)>; // return bool to signal modified/update?
  using vec_predraw_mod_func = std::vector<predraw_mod_func>;
  using vec_postdraw_mod_func = std::vector<postdraw_mod_func>;

  struct Entry {
    ROOT::RDF::RResultPtr<TH1> histo;
    std::string options;
    std::vector<predraw_mod_func> pre_funcs;
    std::vector<postdraw_mod_func> post_funcs;

    Entry& add_pre(predraw_mod_func func) { pre_funcs.push_back(func); return *this;}
    Entry& add_post(postdraw_mod_func func) { post_funcs.push_back(func); return *this; }
    Entry& add_pre(vec_predraw_mod_func &funcs) { pre_funcs.insert(pre_funcs.end(), funcs.begin(), funcs.end()); return *this; }
    Entry& add_post(vec_postdraw_mod_func funcs) { post_funcs.insert(post_funcs.end(), funcs.begin(), funcs.end()); return *this; }
  };

  std::vector<Entry> m_entries;
  TCanvas *m_canvas = nullptr;
  int m_n_divs = -1;

  CanvasGroup(const char *n=0, const char *t=0, const char *pfx=0);
  CanvasGroup(int dx, int dy, const char *n=0, const char *t=0, const char *pfx=0);

  Entry& Add(ROOT::RDF::RResultPtr<TH1> histo, std::string_view opts = "") {
    m_entries.push_back( { histo, std::string(opts), {}, {} } );
    return m_entries.back();
  }

  Entry& Entry() {
    if ( ! m_entries.empty() )
      return m_entries.back();
    throw std::runtime_error("CanvasGroup empty when requesting current entry.");
  }

  CanvasGroup& MakeCanvas(const char *n=0, const char *t=0, const char *pfx=0);
  void Divide(int dx, int dy);
  void DivideSquare(int n = 0);

  void Draw();

  static int s_canvas_width;
  static int s_canvas_height;
  static int s_canvas_counter;

  static const predraw_mod_func stats;
  static const predraw_mod_func logx, logy, logz;

  // Useful?
  // static const postdraw_mod_func post_stats = [](TH1 *h, TVirtualPad *p) { h->SetStats(1); p->Modified(); p->Update(); };
};

typedef CanvasGroup CGrp;

#endif

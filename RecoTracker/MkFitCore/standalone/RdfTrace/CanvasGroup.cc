#include "CanvasGroup.h"

#include "TCanvas.h"

TCanvas *canvas_ptr = nullptr;
int canvas_i = -1, canvas_imax = -1;

void np() {
  if (++canvas_i > canvas_imax) canvas_i = 1;
  canvas_ptr->cd(canvas_i);
}

void nc(int dx, int dy, const char *n, const char *t, const char *pfx) {
  static int n_canvases = 0;
  ++n_canvases;
  TString name, title;
  if (n == 0) {
    name.Form("c_%d", n_canvases);
  } else {
    name = n;
  }
  if (t == 0) {
    title.Form("Canvas %d", n_canvases);
  } else {
    if (pfx)
      title.Form("%s -- %s", t, pfx);
    else
      title = t;
  }
  canvas_ptr = new TCanvas(name, title);
  canvas_imax = dx * dy;
  canvas_i = 0;
  if (dx>1 || dy>1) {
    canvas_ptr->Divide(dx, dy);
    np();
  }
}

// void logx(int f=1) { gPad->SetLogx(f); }
// void logy(int f=1) { gPad->SetLogy(f); }
// void logz(int f=1) { gPad->SetLogz(f); }

// ================================================================

int CanvasGroup::s_canvas_width = 1200;
int CanvasGroup::s_canvas_height = 800;
int CanvasGroup::s_canvas_counter = 0;

const CanvasGroup::predraw_mod_func CanvasGroup::stats = [](TH1 *h, TVirtualPad *p) { h->SetStats(1); };
const CanvasGroup::predraw_mod_func CanvasGroup::logx = [](TH1 *h, TVirtualPad *p) { p->SetLogx(1); };
const CanvasGroup::predraw_mod_func CanvasGroup::logy = [](TH1 *h, TVirtualPad *p) { p->SetLogy(1); };
const CanvasGroup::predraw_mod_func CanvasGroup::logz = [](TH1 *h, TVirtualPad *p) { p->SetLogz(1); };

CanvasGroup::CanvasGroup(const char *n, const char *t, const char *pfx)
{
  MakeCanvas(n, t, pfx);
}

CanvasGroup::CanvasGroup(int dx, int dy, const char *n, const char *t, const char *pfx)
{
  MakeCanvas(n, t, pfx);
  Divide(dx, dy);
}

CanvasGroup& CanvasGroup::MakeCanvas(const char *n, const char *t, const char *pfx) {
  s_canvas_counter++;
  TString name, title;
  if (n == 0) {
    name.Form("cg_%d", s_canvas_counter);
  } else {
    name = n;
  }
  if (t == 0) {
    title.Form("Canvas Group %d", s_canvas_counter);
  } else {
    if (pfx)
      title.Form("%s -- %s", t, pfx);
    else
      title = t;
  }
  m_canvas = new TCanvas(name, title, s_canvas_width, s_canvas_height);
  return *this;
}

void CanvasGroup::Divide(int dx, int dy) {
  if (dx > 1 || dy > 1) {
    m_canvas->Divide(dx, dy);
    m_n_divs = dx * dy;
  } else {
    m_n_divs = 1;
  }
}

void CanvasGroup::DivideSquare(int n) {
  if (n <= 0)
    n = m_entries.size();
  int w = 1, h = 1;
  double nsqrt = std::sqrt((double)n);
  if (m_canvas->GetWindowWidth() > m_canvas->GetWindowHeight()) {
    w = std::ceil(nsqrt);
    h = std::floor(nsqrt);
    if (w*h < n) w++;
  } else {
    h = std::ceil(nsqrt);
    w = std::floor(nsqrt);
    if (w*h < n) h++;
  }
  Divide(w, h);
}

void CanvasGroup::Draw() {
  if ( ! m_canvas)
    MakeCanvas();
  if (m_n_divs < 1)
    DivideSquare();

  for (size_t i = 0; i < m_entries.size(); ++i)
  {
    auto &entry = m_entries[i];
    if ( ! entry.histo ) continue;

    TVirtualPad* pad = m_canvas->cd(i + 1);
    TH1* h_ptr = entry.histo.GetPtr(); 

    for (auto& func : entry.pre_funcs)
      func(h_ptr, pad);

    h_ptr->Draw(entry.options.c_str());

    // ??? This might need to be done AFTER drawin, in another pass
    // Also, will probably need to do pad->Modified(), pad/canvas->Update() after each modification, or at least at the end of all modifications
    // Return from post-func could steer this decision
    for (auto& func : entry.post_funcs) {
      func(h_ptr, pad);
      }
  }
}

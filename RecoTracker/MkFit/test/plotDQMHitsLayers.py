from ROOT import * 
from numpy import array as ar
import os, sys
gROOT.SetBatch()
import subprocess

def main():
    
    if len(sys.argv)<2:
        print("python plotDQMHitsLayers.py DQM_ref.root DQM_1.root ... DQM_4.root")
        exit(0)

    f=[]
    nF=0
    for i in range(1,len(sys.argv)):
        f.append(TFile(sys.argv[i],"OPEN"))
        nF=nF+1
 
    if nF>5:
        print("Max # input files = 5; exiting...")
        exit(0)
        
    plots = [
        "LayersWithMeas_eta",
        "PXBhits_vs_eta",
        "PXFhits_vs_eta",
        "PXLhits_vs_eta",
        "PXLlayersWithMeas_vs_eta",
        "STRIPhits_vs_eta",
        "STRIPlayersWith1dMeas_vs_eta",
        "STRIPlayersWith2dMeas_vs_eta",
        "STRIPlayersWithMeas_vs_eta",
        "TEChits_vs_eta",
        "TIBhits_vs_eta",
        "TIDhits_vs_eta",
        "TOBhits_vs_eta",
    ]

    dirs = [
        "cutsRecoHp_trackingParticleRecoAsssociation",
        "cutsRecoInitialStepByOriginalAlgoHp_trackingParticleRecoAsssociation",
        "cutsRecoHighPtTripletStepByOriginalAlgoHp_trackingParticleRecoAsssociation",
        "cutsRecoDetachedQuadStepByOriginalAlgoHp_trackingParticleRecoAsssociation",
        "cutsRecoDetachedTripletStepByOriginalAlgoHp_trackingParticleRecoAsssociation",
        "cutsRecoLowPtQuadStepByOriginalAlgoHp_trackingParticleRecoAsssociation",
        "cutsRecoLowPtTripletStepByOriginalAlgoHp_trackingParticleRecoAsssociation",
        "cutsRecoMixedTripletStepByOriginalAlgoHp_trackingParticleRecoAsssociation",
        "cutsRecoPixelPairStepByOriginalAlgoHp_trackingParticleRecoAsssociation",
        "cutsRecoPixelLessStepByOriginalAlgoHp_trackingParticleRecoAsssociation",
        "cutsRecoTobTecStepByOriginalAlgoHp_trackingParticleRecoAsssociation",
    ]
    dirs = dirs + [
        "initialStepPreSplitting_quickAssociatorByHitsPreSplitting",
        "initialStep_quickAssociatorByHits",
        "highPtTripletStep_quickAssociatorByHits",
        "detachedQuadStep_quickAssociatorByHits",
        "detachedTripletStep_quickAssociatorByHits",
        "lowPtQuadStep_quickAssociatorByHits",
        "lowPtTripletStep_quickAssociatorByHits",
        "mixedTripletStep_quickAssociatorByHits",
        "pixelPairStep_quickAssociatorByHits",
        "pixelLessStep_quickAssociatorByHits",
        "tobTecStep_quickAssociatorByHits",
    ]

    outdir = "plotsDQM_HitsLayers"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for d in dirs:
        if not os.path.exists("%s/%s"%(outdir,d)):
            os.makedirs("%s/%s"%(outdir,d))
    
    yRanges = {
        "LayersWithMeas_eta" : [0,20],
        "PXBhits_vs_eta" : [0,8],
        "PXFhits_vs_eta" : [0,6],
        "PXLhits_vs_eta" : [0,14],
        "PXLlayersWithMeas_vs_eta" : [0,7],
        "STRIPhits_vs_eta" : [0,25],
        "STRIPlayersWith1dMeas_vs_eta" : [0,10],
        "STRIPlayersWith2dMeas_vs_eta" : [0,10],
        "STRIPlayersWithMeas_vs_eta" : [0,20],
        "TEChits_vs_eta" : [0,20],
        "TIBhits_vs_eta" : [0,10],
        "TIDhits_vs_eta" : [0,10],
        "TOBhits_vs_eta" : [0,10],
    }

    axisTitles = {
        "LayersWithMeas_eta" : ["Track #eta","<# layers>"],
        "PXBhits_vs_eta" : ["Track #eta","<# PXB hits>"],
        "PXFhits_vs_eta" : ["Track #eta","<# PXF hits>"],
        "PXLhits_vs_eta" : ["Track #eta","<# PXL hits>"],
        "PXLlayersWithMeas_vs_eta" : ["Track #eta","<# PXL layers>"],
        "STRIPhits_vs_eta" : ["Track #eta","<# strip hits>"],
        "STRIPlayersWith1dMeas_vs_eta" : ["Track #eta","<# strip layers 1D>"],
        "STRIPlayersWith2dMeas_vs_eta" : ["Track #eta","<# strip layers 2D>"],
        "STRIPlayersWithMeas_vs_eta" : ["Track #eta","<# strip layers>"],
        "TEChits_vs_eta" : ["Track #eta","<# TEC hits>"],
        "TIBhits_vs_eta" : ["Track #eta","<# TIB hits>"],
        "TIDhits_vs_eta" : ["Track #eta","<# TID hits>"],
        "TOBhits_vs_eta" : ["Track #eta","<# TOB hits>"],
    }

    colors = [4,2,1,8,6]
    
    legtit = "t#bar{t} MC with PU"
    
    legtxt = [
        "Reference",
        "Other #1",
        "Other #2",
        "Other #3",
        "Other #4",
    ]

    for d in dirs:
        fulldir="DQMData/Run 1/Tracking/Run summary/Track/%s"%(d)
        if "cutsReco" not in d:
            fulldir="DQMData/Run 1/Tracking/Run summary/TrackBuilding/%s"%(d)
        for hist in plots: 

            canv = TCanvas("c1","c1",800,800)
            canv.cd()
            plotPad = TPad("plotPad","plotPad",0,0.3,1,1)
            ratioPad = TPad("ratioPad","ratioPad",0,0.,1,0.3)
            gStyle.SetOptStat(0)
            plotPad.Draw()
            ratioPad.Draw()
            plotPad.cd()
            plotPad.SetGrid()

            leg = TLegend(0.70, 0.85-0.05*nF, 0.88, 0.88,legtit)
            leg.SetFillColor(10)
            leg.SetFillStyle(0)
            leg.SetLineColor(10)
            leg.SetShadowColor(0)
            leg.SetBorderSize(1)

            h=[]
            r=[]
            den=[]
            for fn in range(nF):
                h.append(f[fn].Get("%s/%s"%(fulldir,hist)))
                h[fn].SetTitle("")
                h[fn].GetXaxis().SetTitle(axisTitles[hist][0])
                h[fn].GetYaxis().SetTitle(axisTitles[hist][1])
                h[fn].Sumw2()
                h[fn].SetMarkerColor(colors[fn])
                h[fn].SetLineColor(colors[fn])
                h[fn].SetLineWidth(2)
                r.append(h[fn].Clone("ratio_%d"%fn))
                den.append(h[0].Clone("denominator_%d"%fn))
                if h[0].ClassName()=="TProfile":
                    for i in range(0,h[0].GetNbinsX()+1):
                        r[fn].SetBinError(i,abs(r[fn].GetBinContent(i))**0.5)
                        den[fn].SetBinError(i,abs(den[fn].GetBinContent(i))**0.5)
                r[fn].Divide(den[fn])
                r[fn].GetXaxis().SetTitle("")
                r[fn].GetYaxis().SetTitle("Ratio")
                r[fn].GetYaxis().SetTitleSize(0.1)
                r[fn].GetYaxis().SetTitleOffset(0.3)
                r[fn].GetXaxis().SetLabelSize(0.06)
                r[fn].GetYaxis().SetLabelSize(0.06)                

                if fn==0:
                    h[fn].GetYaxis().SetRangeUser(yRanges[hist][0],yRanges[hist][1])
                    h[fn].Draw("hist")
                else:
                    h[fn].Draw("hist,same")
                
                leg.AddEntry(h[fn],legtxt[fn],"LP")

            leg.Draw()
            
            ratioPad.cd()
            ratioPad.SetGrid()

            minR=+999.0
            maxR=-999.0
            for fn in range(nF):
                if r[fn].GetMinimum()<minR:
                    minR=r[fn].GetMinimum()
                if r[fn].GetMaximum()>maxR:
                    maxR=r[fn].GetMaximum()
            for fn in range(nF):
                if fn==0:
                    continue
                elif fn==1:
                    r[fn].SetMinimum(minR*0.9)
                    r[fn].SetMaximum(maxR*0.9)
                    r[fn].GetYaxis().SetRangeUser(minR*0.9,maxR*1.1)
                    r[fn].Draw("hist")
                else:
                    r[fn].Draw("hist,same")
            line = TLine(r[0].GetXaxis().GetBinLowEdge(1),1,r[0].GetXaxis().GetBinUpEdge(r[0].GetNbinsX()),1)
            line.SetLineColor(colors[0])
            line.SetLineStyle(kDashed)
            line.Draw("same")

            plotPad.RedrawAxis()
            canv.SaveAs("%s/%s/%s.pdf"%(outdir,d,hist))
            canv.SaveAs("%s/%s/%s.png"%(outdir,d,hist))

        subprocess.Popen("cp ${MKFIT_BASE}/web/index.php %s/%s"%(outdir,d),shell=True)
    subprocess.Popen("cp ${MKFIT_BASE}/web/index.php %s"%(outdir),shell=True)

main()

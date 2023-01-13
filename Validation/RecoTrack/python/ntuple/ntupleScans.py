import math
# match from t to tO(ther)
def scanTC(t, tO, minSimPt=0., maxSimPt = 1e33, nEv=10, iteration=9, minSimFrac=0.75, pSeeds = True, pCHits = True):
  nGood = 0
  nNotMatched2O = 0
  for iE in range(t.GetEntries()):
    nGood_Ev = 0
    nNotMatched2O_Ev = 0
    if iE >= nEv: continue
    nb = t.GetEntry(iE)
    nb = tO.GetEntry(iE)
    tAlg = t.tcand_algo
    tAlgO = tO.tcand_algo
    tSim = t.tcand_bestSimTrkIdx
    tSimO = tO.tcand_bestSimTrkIdx
    # list of cand idx,simIdx with a match and iteration
    lSel = [(i,v) for i,v in enumerate(tSim) if v>=0 and tAlg[i]==iteration]
    lSelO = [(i,v) for i,v in enumerate(tSimO) if v>=0 and tAlgO[i]==iteration]
    # list of simIdx that have a cand in the selected iteration
    simsSel = [v for i,v in lSel]
    simsSelO = [v for i,v in lSelO]
    for i,iSim in lSel:
      # select good cand
      for iT in t.sim_trkIdx[iSim]:
        # there is no direct map to tcand except from see
        iST = t.trk_seedIdx[iT]
        if t.see_tcandIdx[iST] != i: continue
        if not t.trk_isHP[iT]: continue
        if t.tcand_bestSimTrkShareFrac[i] < minSimFrac: continue
        if t.sim_pt[iSim] < minSimPt or t.sim_pt[iSim] > maxSimPt: continue
        nGood = nGood + 1
        nGood_Ev = nGood_Ev + 1
      
        if iSim not in simsSelO:
          print(iE,"tc ",i," sim ",iSim," not found in Other")
          nNotMatched2O = nNotMatched2O + 1
          nNotMatched2O_Ev = nNotMatched2O_Ev + 1
          
          # get seed label in a given iteration
          iLab = -1
          algLast = -1
          for iis,alg in enumerate(t.see_algo):
            if alg != algLast:
              algLast = alg
              iLab = -1
            iLab = iLab + 1
            if iis == iST: break
            
          if pSeeds: print(f'see {iST:3d} {iLab:3d} {t.see_pt[iST]: 6.4f} {t.see_eta[iST]: 6.4f} {t.see_phi[iST]: 6.4f} {t.see_nValid[iST]:2d} {t.see_stateTrajGlbX[iST]: 6.4f} {t.see_stateTrajGlbY[iST]: 6.4f} {t.see_stateTrajGlbZ[iST]: 6.4f} {t.see_stateTrajGlbPx[iST]: 6.4f} {t.see_stateTrajGlbPy[iST]: 6.4f} {t.see_stateTrajGlbPz[iST]: 6.4f}')
          for iSO in (iiSO for iiSO in t.sim_seedIdx[iSim] if iiSO!=iST):
            if pSeeds: print(f'seeO {iSO:3d} {t.see_pt[iSO]: 6.4f} {t.see_eta[iSO]: 6.4f} {t.see_phi[iSO]: 6.4f} {t.see_nValid[iSO]:2d} {t.see_stateTrajGlbX[iSO]: 6.4f} {t.see_stateTrajGlbY[iSO]: 6.4f} {t.see_stateTrajGlbZ[iSO]: 6.4f} {t.see_stateTrajGlbPx[iSO]: 6.4f} {t.see_stateTrajGlbPy[iSO]: 6.4f} {t.see_stateTrajGlbPz[iSO]: 6.4f}')
          sehs = []
          for (ih,ht) in enumerate(t.see_hitType[iST]):
            iS = t.see_hitIdx[iST][ih]
            sehs.append(iS)
            if (ht == 0) and pSeeds: print(f'P {iS:4d} {t.pix_subdet[iS]: 2d} {t.pix_layer[iS]: 3d} ({t.pix_x[iS]: 6.2f} {t.pix_y[iS]: 6.2f} {t.pix_z[iS]: 6.2f}) cm {sh: 4d}')
            if (ht == 1) and pSeeds: print(f'S {iS:4d} {t.str_subdet[iS]:2d} {t.str_layer[iS]:3d} {t.str_isStereo[iS]:2d} ({t.str_x[iS]: 6.2f} {t.str_y[iS]: 6.2f} {t.str_z[iS]: 6.2f}) cm')
            if (ht == 2) and pSeeds:
              sh=t.glu_stereoIdx[iS]
              mh=t.glu_monoIdx[iS]
              sehs.append(sh)
              sehs.append(mh)
              print(f'g {iS:4d} {t.glu_subdet[iS]:2d} {t.glu_layer[iS]:3d} ({t.glu_x[iS]: 6.2f} {t.glu_y[iS]: 6.2f} {t.glu_z[iS]: 6.2f}) cm {t.glu_xx[iS]: 6.2e} {t.glu_xy[iS]: 6.2e} {t.glu_yy[iS]: 6.2e} m {mh:4d} ({t.str_x[mh]: 6.2f} {t.str_y[mh]: 6.2f} {t.str_z[mh]: 6.2f}) s {sh:4d} ({t.str_x[sh]: 6.2f} {t.str_y[sh]: 6.2f} {t.str_z[sh]: 6.2f})')
          print(f'tcand {i:4d} {iST:4d} {t.tcand_x[i]: 6.4f} {t.tcand_y[i]: 6.4f} {t.tcand_pca_pt[i]:6.4f} {t.tcand_pca_eta[i]: 6.4f} {t.tcand_pca_phi[i]: 6.4f} {t.tcand_nValid[i]:2d} {t.tcand_nPixel[i]:2d} {t.tcand_bestSimTrkIdx[i]:3d} {t.sim_pdgId[iSim]: 5d} {t.sim_parentVtxIdx[iSim]:3d} {t.tcand_bestSimTrkShareFrac[i]:4.3f} {t.sim_pt[iSim]:6.4f} {t.sim_eta[iSim]: 6.4f}')
          tchs = []
          iSimHits = t.sim_simHitIdx[iSim]
          for (ih,ht) in enumerate(t.tcand_hitType[i]):
            iS = t.tcand_hitIdx[i][ih]
            sehSt = "SH" if iS in sehs else ""
            tchs.append([ht,iS])
            if (ht == 0) and pCHits:
              shs = [s for s in t.pix_simHitIdx[iS] if s in iSimHits]
              sh = -1 if len(shs)==0 else shs[0]
              shr = [0, 0, 0] if sh == -1 else [t.simhit_x[sh], t.simhit_y[sh], t.simhit_z[sh]]
              shp = [0, 0, 0] if sh == -1 else [t.simhit_particle[sh], math.sqrt(t.simhit_px[sh]*t.simhit_px[sh]+t.simhit_py[sh]*t.simhit_py[sh]), t.simhit_pz[sh]]
              print(f'P {iS:4d} {t.pix_subdet[iS]: 2d} {t.pix_layer[iS]: 3d} ({t.pix_x[iS]: 6.2f} {t.pix_y[iS]: 6.2f} {t.pix_z[iS]: 6.2f}) cm {sh: 4d} ({shr[0]: 6.2f} {shr[1]: 6.2f} {shr[2]: 6.2f}) {shp[0]: 5d} ({shp[1]: 6.2f} {shp[2]: 6.2f}) {sehSt}')
            if (ht == 1) and pCHits:
              shs = [s for s in t.str_simHitIdx[iS] if s in iSimHits]
              sh = -1 if len(shs)==0 else shs[0]
              shr = [0, 0, 0] if sh == -1 else [t.simhit_x[sh], t.simhit_y[sh], t.simhit_z[sh]]
              shp = [0, 0, 0] if sh == -1 else [t.simhit_particle[sh], math.sqrt(t.simhit_px[sh]*t.simhit_px[sh]+t.simhit_py[sh]*t.simhit_py[sh]), t.simhit_pz[sh]]
              print(f'S {iS:4d} {t.str_subdet[iS]:2d} {t.str_layer[iS]:3d} {t.str_isStereo[iS]:2d} ({t.str_x[iS]: 6.2f} {t.str_y[iS]: 6.2f} {t.str_z[iS]: 6.2f}) cm {sh:4d}  ({shr[0]: 6.2f} {shr[1]: 6.2f} {shr[2]: 6.2f}) {shp[0]: 5d} ({shp[1]: 6.2f} {shp[2]: 6.2f}) {sehSt}')
          for sh in iSimHits:
            shr = [t.simhit_x[sh], t.simhit_y[sh], t.simhit_z[sh]]
            shp = [t.simhit_particle[sh], math.sqrt(t.simhit_px[sh]*t.simhit_px[sh]+t.simhit_py[sh]*t.simhit_py[sh]), t.simhit_pz[sh]]
            if (shp[1]<0.05): continue
            for (ih,ht) in enumerate(t.simhit_hitType[sh]):
              iS = t.simhit_hitIdx[sh][ih]
              if ([ht,iS] in tchs): continue
              if (ht == 0) and pCHits:
                print(f'P {iS:4d} {t.pix_subdet[iS]: 2d} {t.pix_layer[iS]: 3d} ({t.pix_x[iS]: 6.2f} {t.pix_y[iS]: 6.2f} {t.pix_z[iS]: 6.2f}) cm {sh: 4d} ({shr[0]: 6.2f} {shr[1]: 6.2f} {shr[2]: 6.2f}) {shp[0]: 5d} ({shp[1]: 6.2f} {shp[2]: 6.2f})  extra')
              if (ht == 1) and pCHits:
                print(f'S {iS:4d} {t.str_subdet[iS]:2d} {t.str_layer[iS]:3d} {t.str_isStereo[iS]:2d} ({t.str_x[iS]: 6.2f} {t.str_y[iS]: 6.2f} {t.str_z[iS]: 6.2f}) cm {sh:4d}  ({shr[0]: 6.2f} {shr[1]: 6.2f} {shr[2]: 6.2f}) {shp[0]: 5d} ({shp[1]: 6.2f} {shp[2]: 6.2f}) extra')
    if nGood_Ev > 0: print(iE,"summary: nGood",nGood_Ev,"missed",nNotMatched2O_Ev,f"frac {nNotMatched2O_Ev/nGood_Ev:6.3f}")
    for i,iSim in lSelO:
      if iSim not in simsSel:
        # a placeholder, effectively disabled
        if tO.sim_trkIdx[iSim].size()>100:
          print(iE,"tc ",i," sim ",iSim," only in tO(ther)")
  if nGood > 0: print("summary for ",nEv,": nGood",nGood,"missed",nNotMatched2O,f"frac {nNotMatched2O/nGood:6.3f}")

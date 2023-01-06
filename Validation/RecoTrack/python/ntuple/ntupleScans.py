import math
def scanTC(ta, tb, minSimPt=0., nEv=10, iteration=9, minSimFrac=0.75):
  for iE in range(ta.GetEntries()):
    if iE > nEv: continue
    nb = ta.GetEntry(iE)
    nb = tb.GetEntry(iE)
    taa = ta.tcand_algo
    tab = tb.tcand_algo
    tca = ta.tcand_bestSimTrkIdx
    tcb = tb.tcand_bestSimTrkIdx
    l9a = [(i,v) for i,v in enumerate(tca) if v>=0 and taa[i]==iteration]
    l9b = [(i,v) for i,v in enumerate(tcb) if v>=0 and tab[i]==iteration]
    v9a = [v for i,v in l9a]
    v9b = [v for i,v in l9b]
    for i,v in l9a:
      if v not in v9b:
        for iT in ta.sim_trkIdx[v]:
          # there is no direct map to tcand except from see
          iST = ta.trk_seedIdx[iT]
          if ta.see_tcandIdx[iST] != i: continue
          if not ta.trk_isHP[iT]: continue
          if ta.tcand_bestSimTrkShareFrac[i] < minSimFrac: continue
          t = ta
          if t.sim_pt[v] < minSimPt: continue
          print(iE,"tc ",i," sim ",v," only in a")
          algLast = -1
          iLab = -1
          for iis,alg in enumerate(ta.see_algo):
            if alg != algLast:
              algLast = alg
              iLab = -1
            iLab = iLab + 1
            if iis == iST: break
          print(f'see {iST:3d} {iLab:3d} {t.see_pt[iST]: 6.4f} {t.see_eta[iST]: 6.4f} {t.see_phi[iST]: 6.4f} {t.see_nValid[iST]:2d} {t.see_stateTrajGlbX[iST]: 6.4f} {t.see_stateTrajGlbY[iST]: 6.4f} {t.see_stateTrajGlbZ[iST]: 6.4f} {t.see_stateTrajGlbPx[iST]: 6.4f} {t.see_stateTrajGlbPy[iST]: 6.4f} {t.see_stateTrajGlbPz[iST]: 6.4f}')
          for iSO in (iiSO for iiSO in t.sim_seedIdx[v] if iiSO!=iST):
            print(f'seeO {iSO:3d} {t.see_pt[iSO]: 6.4f} {t.see_eta[iSO]: 6.4f} {t.see_phi[iSO]: 6.4f} {t.see_nValid[iSO]:2d} {t.see_stateTrajGlbX[iSO]: 6.4f} {t.see_stateTrajGlbY[iSO]: 6.4f} {t.see_stateTrajGlbZ[iSO]: 6.4f} {t.see_stateTrajGlbPx[iSO]: 6.4f} {t.see_stateTrajGlbPy[iSO]: 6.4f} {t.see_stateTrajGlbPz[iSO]: 6.4f}')
          sehs = []
          for (ih,ht) in enumerate(t.see_hitType[iST]):
            iS = t.see_hitIdx[iST][ih]
            sehs.append(iS)
            if (ht == 0): print(f'P {iS:4d} {t.pix_subdet[iS]: 2d} {t.pix_layer[iS]: 3d} ({t.pix_x[iS]: 6.2f} {t.pix_y[iS]: 6.2f} {t.pix_z[iS]: 6.2f}) cm {sh: 4d}')
            if (ht == 1): print(f'S {iS:4d} {t.str_subdet[iS]:2d} {t.str_layer[iS]:3d} {t.str_isStereo[iS]:2d} ({t.str_x[iS]: 6.2f} {t.str_y[iS]: 6.2f} {t.str_z[iS]: 6.2f}) cm')
            if (ht == 2):
              sh=t.glu_stereoIdx[iS]
              mh=t.glu_monoIdx[iS]
              sehs.append(sh)
              sehs.append(mh)
              print(f'g {iS:4d} {t.glu_subdet[iS]:2d} {t.glu_layer[iS]:3d} ({t.glu_x[iS]: 6.2f} {t.glu_y[iS]: 6.2f} {t.glu_z[iS]: 6.2f}) cm {t.glu_xx[iS]: 6.2e} {t.glu_xy[iS]: 6.2e} {t.glu_yy[iS]: 6.2e} m {mh:4d} ({t.str_x[mh]: 6.2f} {t.str_y[mh]: 6.2f} {t.str_z[mh]: 6.2f}) s {sh:4d} ({t.str_x[sh]: 6.2f} {t.str_y[sh]: 6.2f} {t.str_z[sh]: 6.2f})')
          print(f'tcand {i:4d} {iST:4d} {t.tcand_x[i]: 6.4f} {t.tcand_y[i]: 6.4f} {t.tcand_pca_pt[i]:6.4f} {t.tcand_pca_eta[i]: 6.4f} {t.tcand_pca_phi[i]: 6.4f} {t.tcand_nValid[i]:2d} {t.tcand_nPixel[i]:2d} {t.tcand_bestSimTrkIdx[i]:3d} {t.sim_pdgId[v]: 5d} {t.sim_parentVtxIdx[v]:3d} {t.tcand_bestSimTrkShareFrac[i]:4.3f} {t.sim_pt[v]:6.4f} {t.sim_eta[v]: 6.4f}')
          tchs = []
          for (ih,ht) in enumerate(t.tcand_hitType[i]):
            iS = t.tcand_hitIdx[i][ih]
            sehSt = "SH" if iS in sehs else ""
            tchs.append([ht,iS])
            if (ht == 0):
              shs = [s for s in t.pix_simHitIdx[iS] if s in t.sim_simHitIdx[v]]
              sh = -1 if len(shs)==0 else shs[0]
              shr = [0, 0, 0] if sh == -1 else [t.simhit_x[sh], t.simhit_y[sh], t.simhit_z[sh]]
              shp = [0, 0, 0] if sh == -1 else [t.simhit_particle[sh], math.sqrt(t.simhit_px[sh]*t.simhit_px[sh]+t.simhit_py[sh]*t.simhit_py[sh]), t.simhit_pz[sh]]
              print(f'P {iS:4d} {t.pix_subdet[iS]: 2d} {t.pix_layer[iS]: 3d} ({t.pix_x[iS]: 6.2f} {t.pix_y[iS]: 6.2f} {t.pix_z[iS]: 6.2f}) cm {sh: 4d} ({shr[0]: 6.2f} {shr[1]: 6.2f} {shr[2]: 6.2f}) {shp[0]: 5d} ({shp[1]: 6.2f} {shp[2]: 6.2f}) {sehSt}')
            if (ht == 1):
              shs = [s for s in t.str_simHitIdx[iS] if s in t.sim_simHitIdx[v]]
              sh = -1 if len(shs)==0 else shs[0]
              shr = [0, 0, 0] if sh == -1 else [t.simhit_x[sh], t.simhit_y[sh], t.simhit_z[sh]]
              shp = [0, 0, 0] if sh == -1 else [t.simhit_particle[sh], math.sqrt(t.simhit_px[sh]*t.simhit_px[sh]+t.simhit_py[sh]*t.simhit_py[sh]), t.simhit_pz[sh]]
              print(f'S {iS:4d} {t.str_subdet[iS]:2d} {t.str_layer[iS]:3d} {t.str_isStereo[iS]:2d} ({t.str_x[iS]: 6.2f} {t.str_y[iS]: 6.2f} {t.str_z[iS]: 6.2f}) cm {sh:4d}  ({shr[0]: 6.2f} {shr[1]: 6.2f} {shr[2]: 6.2f}) {shp[0]: 5d} ({shp[1]: 6.2f} {shp[2]: 6.2f}) {sehSt}')
          for sh in t.sim_simHitIdx[v]:
            shr = [t.simhit_x[sh], t.simhit_y[sh], t.simhit_z[sh]]
            shp = [t.simhit_particle[sh], math.sqrt(t.simhit_px[sh]*t.simhit_px[sh]+t.simhit_py[sh]*t.simhit_py[sh]), t.simhit_pz[sh]]
            if (shp[1]<0.05): continue
            for (ih,ht) in enumerate(t.simhit_hitType[sh]):
              iS = t.simhit_hitIdx[sh][ih]
              if ([ht,iS] in tchs): continue
              if (ht == 0):
                print(f'P {iS:4d} {t.pix_subdet[iS]: 2d} {t.pix_layer[iS]: 3d} ({t.pix_x[iS]: 6.2f} {t.pix_y[iS]: 6.2f} {t.pix_z[iS]: 6.2f}) cm {sh: 4d} ({shr[0]: 6.2f} {shr[1]: 6.2f} {shr[2]: 6.2f}) {shp[0]: 5d} ({shp[1]: 6.2f} {shp[2]: 6.2f})  extra')
              if (ht == 1):
                print(f'S {iS:4d} {t.str_subdet[iS]:2d} {t.str_layer[iS]:3d} {t.str_isStereo[iS]:2d} ({t.str_x[iS]: 6.2f} {t.str_y[iS]: 6.2f} {t.str_z[iS]: 6.2f}) cm {sh:4d}  ({shr[0]: 6.2f} {shr[1]: 6.2f} {shr[2]: 6.2f}) {shp[0]: 5d} ({shp[1]: 6.2f} {shp[2]: 6.2f}) extra')
    for i,v in l9b:
      if v not in v9a:
        # a placeholder, effectively disabled
        if tb.sim_trkIdx[v].size()>100:
          print(iE,"tc ",i," sim ",v," only in B")

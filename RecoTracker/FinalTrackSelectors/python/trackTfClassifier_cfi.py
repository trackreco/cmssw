from Configuration.ProcessModifiers.trackdnn_CKF_cff import trackdnn_CKF
import RecoTracker.FinalTrackSelectors.trackTfClassifierDefault_cfi as _mod
import RecoTracker.FinalTrackSelectors.trackTfClassifierDefaultBatch_cfi as _modb

trackTfClassifier = _mod.trackTfClassifierDefault.clone()
trackTfClassifierBatch = _modb.trackTfClassifierDefaultBatch.clone()

trackdnn_CKF.toModify(trackTfClassifier.mva, tfDnnLabel = 'trackSelectionTf_CKF')

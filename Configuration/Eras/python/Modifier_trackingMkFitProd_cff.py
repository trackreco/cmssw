import FWCore.ParameterSet.Config as cms

# to replace CKF with MkFit in select iterations
# to be renamed to 'trackingMkFit' in a calmer time period to avoid merge conflicts with other developments
trackingMkFitProd =  cms.Modifier()

import numpy as np

class Feature_Extractor:
    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_feature_list(self):
        feature_list = ['MAV',
                        'ZC',
                        'SSC',
                        'WL',
                        'LS',
                        'MFL',
                        'MSR',
                        'WAMP',
                        'RMS',
                        'IAV',
                        'DASDV',
                        'VAR',
                        'M0',
                        'M2',
                        'M4',
                        'SPARSI',
                        'IRF',
                        'WLF']
        return feature_list

    def extract(self, feature_list, windows):
        features = {}
        for feature in feature_list:
            method_to_call = getattr(self, 'get' + feature + 'feat')
            features[feature] = method_to_call(windows)
        return features

    def getMAVfeat(self, windows):
        feat = np.mean(np.abs(windows),2)
        return feat
    
    def getZCfeat(self, windows):
        sgn_change = np.diff(np.sign(windows),axis=2)
        neg_change = sgn_change == -2
        pos_change = sgn_change ==  2
        feat_a = np.sum(neg_change,2)
        feat_b = np.sum(pos_change,2)
        return feat_a+feat_b
    
    def getSSCfeat(self, windows):
        d_sig = np.diff(windows,axis=2)
        return self.getZCfeat(d_sig)

    def getWLfeat(self, windows):
        feat = np.sum(np.abs(np.diff(windows,axis=2)),2)
        return feat

    def getLSfeat(self, windows):
        feat = np.zeros((windows.shape[0],windows.shape[1]))
        for w in range(0, windows.shape[0],1):
            for c in range(0, windows.shape[1],1):
                tmp = self.lmom(np.reshape(windows[w,c,:],(1,windows.shape[2])),2)
                feat[w,c] = tmp[0,1]
        return feat

    def lmom(self, signal, nL):
        # same output to matlab when ones vector of various sizes are input
        b = np.zeros((1,nL-1))
        l = np.zeros((1,nL-1))
        b0 = np.zeros((1,1))
        b0[0,0] = np.mean(signal)
        n = signal.shape[1]
        signal = np.sort(signal, axis=1)
        for r in range(1,nL,1):
            num = np.tile(np.asarray(range(r+1,n+1)),(r,1))  - np.tile(np.asarray(range(1,r+1)),(1,n-r))
            num = np.prod(num,axis=0)
            den = np.tile(np.asarray(n),(1,r)) - np.asarray(range(1,r+1))
            den = np.prod(den)
            b[r-1] = 1/n * np.sum(num / den * signal[0,r:n])
        tB = np.concatenate((b0,b))
        B = np.flip(tB,0)
        for i in range(1, nL, 1):
            Spc = np.zeros((B.shape[0]-(i+1),1))
            Coeff = np.concatenate((Spc, self.LegendreShiftPoly(i)))
            l[0,i-1] = np.sum(Coeff * B)
        L = np.concatenate((b0, l),1)

        return L

    def LegendreShiftPoly(self, n):
        # Verified: this has identical function to MATLAB function for n = 2:10 (only 2 is used to compute LS feature)
        pk = np.zeros((n+1,1))
        if n == 0:
            pk = 1
        elif n == 1:
            pk[0,0] = 2
            pk[1,0] = -1
        else:
            pkm2 = np.zeros(n+1)
            pkm2[n] = 1
            pkm1 = np.zeros(n+1)
            pkm1[n] = -1
            pkm1[n-1] = 2

            for k in range(2,n+1,1):
                pk = np.zeros((n+1,1))
                for e in range(n-k+1,n+1,1):
                    pk[e-1] = (4*k-2)*pkm1[e]+ (1-2*k)*pkm1[e-1] + (1-k) * pkm2[e-1]
                pk[n,0] = (1-2*k)*pkm1[n] + (1-k)*pkm2[n]
                pk = pk/k

                if k < n:
                    pkm2 = pkm1
                    pkm1 = pk

        return pk

    def getMFLfeat(self, windows):
        feat = np.log10(np.sum(np.abs(np.diff(windows, axis=2)),axis=2))
        return feat

    def getMSRfeat(self, windows):
        feat = np.abs(np.mean(np.sqrt(windows.astype('complex')),axis=2))
        return feat

    def getWAMPfeat(self, windows, threshold=2e-3): # TODO: add optimization if threshold not passed, need class labels
        feat = np.sum(np.abs(np.diff(windows, axis=2)) > threshold, axis=2)
        return feat

    def getRMSfeat(self, windows):
        feat = np.sqrt(np.mean(np.square(windows),2))
        return feat

    def getIAVfeat(self, windows):
        feat = np.sum(np.abs(windows),axis=2)
        return feat

    def getDASDVfeat(self, windows):
        feat = np.abs(np.sqrt(np.mean(np.diff(np.square(windows.astype('complex')),2),2)))
        return feat

    def getVARfeat(self, windows):
        feat = np.var(windows,axis=2)
        return feat

    def getM0feat(self, windows):
        # There are 6 features per channel
        m0 = np.sqrt(np.sum(windows**2,axis=2))
        m0 = m0 ** 0.1 / 0.1
        #Feature extraction goes here
        return np.log(np.abs(m0))
    
    def getM2feat(self, windows):
        # Prepare derivatives for higher order moments
        d1 = np.diff(windows, n=1, axis=2)
        # Root squared 2nd order moments normalized
        m2 = np.sqrt(np.sum(d1 **2, axis=2)/ (windows.shape[2]-1))
        m2 = m2 ** 0.1 / 0.1
        return np.log(np.abs(m2))

    def getM4feat(self, windows):
        # Prepare derivatives for higher order moments
        d1 = np.diff(windows, n=1, axis=2)
        d2 = np.diff(d1     , n=1, axis=2)
        # Root squared 4th order moments normalized
        m4 = np.sqrt(np.sum(d2**2,axis=2) / (windows.shape[2]-1))
        m4 = m4 **0.1/0.1
        return np.log(np.abs(m4))
    
    def getSPARSIfeat(self, windows):
        m0 = self.getM0feat(windows)
        m2 = self.getM2feat(windows)
        m4 = self.getM4feat(windows)
        sparsi = m0/np.sqrt(np.abs((m0-m2)*(m0-m4)))
        return np.log(np.abs(sparsi))

    def getIRFfeat(self, windows):
        m0 = self.getM0feat(windows)
        m2 = self.getM2feat(windows)
        m4 = self.getM4feat(windows)
        IRF = m2/np.sqrt(np.multiply(m0,m4))
        return np.log(np.abs(IRF))

    def getWLFfeat(self, windows):
        # Prepare derivatives for higher order moments
        d1 = np.diff(windows, n=1, axis=2)
        d2 = np.diff(d1     , n=1, axis=2)
        # Waveform Length Ratio
        WLR = np.sum( np.abs(d1),axis=2)-np.sum(np.abs(d2),axis=2)
        return np.log(np.abs(WLR))
    # def getTDPSDfeat(self, windows):
    #     # There are 6 features per channel
    #     features = np.zeros((windows.shape[0], self.num_channels*6), dtype=float)
    #     # TDPSD feature set adapted from: https://github.com/RamiKhushaba/getTDPSDfeat
    #     # Extract the features from original signal and nonlinear version
    #     ebp = self.KSM1(windows)
    #     # np.spacing = epsilon (smallest value), done so log does not return inf.
    #     efp = self.KSM1(np.log(windows**2 + np.spacing(1)))
    #     # Correlation analysis:
    #     num = -2*np.multiply(efp, ebp)
    #     den = np.multiply(efp, efp) + np.multiply(ebp,ebp)
    #     #Feature extraction goes here
    #     features = num-den
    #     return features

    # def KSM1(self, signals):
    #     samples = signals.shape[2]
    #     channels = signals.shape[1]
    #     # Root squared zero moment normalized
    #     m0 = np.sqrt(np.sum(signals**2,axis=2))
    #     m0 = m0 ** 0.1 / 0.1
    #     # Prepare derivatives for higher order moments
    #     d1 = np.diff(signals, n=1, axis=2)
    #     d2 = np.diff(d1     , n=1, axis=2)
    #     # Root squared 2nd and 4th order moments normalized
    #     m2 = np.sqrt(np.sum(d1 **2, axis=2)/ (samples-1))
    #     m2 = m2 ** 0.1 / 0.1
    #     m4 = np.sqrt(np.sum(d2**2,axis=2) / (samples-1))
    #     m4 = m4 **0.1/0.1

    #     # Sparseness
    #     sparsi = m0/np.sqrt(np.abs((m0-m2)*(m0-m4)))

    #     # Irregularity factor
    #     IRF = m2/np.sqrt(np.multiply(m0,m4))

    #     # Waveform Length Ratio
    #     WLR = np.sum( np.abs(d1),axis=2)-np.sum(np.abs(d2),axis=2)

    #     Feat = np.concatenate((m0, m0-m2, m0-m4, sparsi, IRF, WLR), axis=1)
    #     Feat = np.log(np.abs(Feat))
    #     return Feat
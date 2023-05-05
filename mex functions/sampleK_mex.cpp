#include "mex.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>

// [Kal,SumX,SumX2,n_ii] = sampleK(Kal,estar,psr,m,tau,s,S,alpha)
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    // Check the number of input and output arguments
    if (nrhs != 8 || nlhs != 4)
    {
        mexErrMsgIdAndTxt("sampleK:nargin", "Invalid number of input/output arguments.");
        return;
    }
    // Get the input arguments
    double* Kal = mxGetPr(prhs[0]);
    double* estar = mxGetPr(prhs[1]);
    double* psr = mxGetPr(prhs[2]);
    double m = mxGetScalar(prhs[3]);
    double tau = mxGetScalar(prhs[4]);
    double s = mxGetScalar(prhs[5]);
    double S = mxGetScalar(prhs[6]);
    double alpha = mxGetScalar(prhs[7]);
    mwSize T = mxGetNumberOfElements(prhs[0]);
    
    // Compute the initial value of k by counting unique clusters in Kal
    std::set<double> unique_clusters(Kal, Kal + T);
    int k = unique_clusters.size();
    
    // Create output variables
    plhs[0] = mxDuplicateArray(prhs[0]);
    double* KalOut = mxGetPr(plhs[0]);
    
    
    //     Initialize n_ii, SumX, and SumX2
    std::vector<double> n_ii(k, 0.0);
    std::vector<double> SumX(k, 0.0);
    std::vector<double> SumX2(k, 0.0);
    
    for (mwSize i = 0; i < T; ++i)
    {
        double xi = estar[i];
        double c_ii = KalOut[i];
        n_ii[c_ii - 1] += 1;
        SumX[c_ii - 1] += xi;
        SumX2[c_ii - 1] += xi * xi;
    }
//     
    // Define pi and pre-allocate some computations
    const double M_PI = std::acos(-1.0);
    double nu_a = 2.0 * s;
    double sig2_a = S * (1.0 + tau) / s;
    double lgamma_nu_a_prior = std::lgamma((nu_a + 1.0) / 2.0) - std::lgamma(nu_a / 2.0);
    double log_nu_a_pi_sig2_a = std::log(nu_a * M_PI * sig2_a);
//     
     
    std::vector<double> nu(k);
    std::vector<double> sig2(k);
    std::vector<double> logintk(k); 
    std::vector<double> Vn(k);
    std::vector<double> mn(k);
    std::vector<double> an(k);
    std::vector<double> bn(k);
    std::vector<double> pi_k(k + 1);
//     
// 
// Iterate over each element of KalOut, potentially updating the cluster allocation
    for (mwSize t = 0; t < T; ++t)
    {
        double c_ii = KalOut[t];
        double xi = estar[t];
        n_ii[c_ii - 1] -= 1;
        SumX[c_ii - 1] -= xi;
        SumX2[c_ii - 1] -= xi * xi;
        
        // If c_ii is not associated with any cluster but itself
        if (n_ii[c_ii - 1] == 0)
        {
            n_ii.erase(n_ii.begin() + (c_ii - 1));
            SumX.erase(SumX.begin() + (c_ii - 1));
            SumX2.erase(SumX2.begin() + (c_ii - 1));
            for (mwSize i = 0; i < T; ++i)
            {
                if (KalOut[i] > c_ii)
                {
                    KalOut[i] -= 1;
                }
            }
            k -= 1; 
            
            // Resize the vectors based on the new value of k 
            nu.resize(k);
            sig2.resize(k);
            logintk.resize(k);
            Vn.resize(k);
            mn.resize(k);
            an.resize(k);
            bn.resize(k);
            pi_k.resize(k + 1);
        }
        
        

        
        // Compute the posteriors
        for (size_t i = 0; i < k; ++i)
        {
            // Compute the categorical distribution 
            Vn[i] = 1.0 / (1.0 / tau + n_ii[i]);
            mn[i] = Vn[i] * (m / tau + SumX[i]);
            an[i] = s + n_ii[i] / 2.0;
            bn[i] = S + 0.5 * (m * m / tau + SumX2[i] - mn[i] * mn[i] / Vn[i]); 
            nu[i] = 2.0 * an[i];
            sig2[i] = bn[i] * (1.0 + Vn[i]) / an[i];
            logintk[i] = std::lgamma((nu[i] + 1.0) / 2.0) - std::lgamma(nu[i] / 2.0)
            - 0.5 * std::log(nu[i] * M_PI * sig2[i])
            - (nu[i] + 1.0) / 2.0 * std::log(1.0 + 1.0 / nu[i] * std::pow((xi - mn[i]), 2.0) / sig2[i]);
            pi_k[i] = n_ii[i] * std::exp(logintk[i]);
        }
// //         
        double logint1 = lgamma_nu_a_prior
                - 0.5 * log_nu_a_pi_sig2_a
                - (nu_a + 1.0) / 2.0 * std::log(1.0 + 1.0 / nu_a * std::pow((xi - m), 2.0) / sig2_a);
        
        pi_k[k] = alpha * std::exp(logint1);
//         
        double total = 0.0;
        for (double val : pi_k)
        {
            total += val;
        }
        for (size_t i = 0; i < pi_k.size(); ++i)
        {
            pi_k[i] /= total;
        }
//         
        // Draw from the categorical distribution using random numbers psr
        double cumulativeProb = 0.0;
        double randNum = psr[t];
        double new_c_ii = 0;
        for (size_t i = 0; i < pi_k.size(); ++i)
        {
            cumulativeProb += pi_k[i];
            if (randNum < cumulativeProb)
            {
                new_c_ii = i + 1;
                break;
            }
        }
        
        //         double new_c_ii = Kal[t]; 
//         new_c_ii = Kal[t];
        KalOut[t] = new_c_ii;

        // If Indicator is new, add a parameter to the cluster locations
        if (new_c_ii == k + 1) // add a new cluster
        {
            n_ii.push_back(1);
            SumX.push_back(xi);
            SumX2.push_back(xi * xi);
            k += 1; // adjust number of clusters 
            nu.resize(k);
            sig2.resize(k);
            logintk.resize(k);
            Vn.resize(k);
            mn.resize(k);
            an.resize(k);
            bn.resize(k);
            pi_k.resize(k + 1);
        }
        else
        {
            n_ii[new_c_ii - 1] += 1;
            SumX[new_c_ii - 1] += xi;
            SumX2[new_c_ii - 1] += xi * xi;
        }
    }
//     
    // Copy the values from n_ii, SumX, and SumX2 vectors to their respective mxArray objects
    // [Kal,SumX,SumX2,n_ii]
    plhs[1] = mxCreateDoubleMatrix(1, SumX.size(), mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1, SumX2.size(), mxREAL);
    plhs[3] = mxCreateDoubleMatrix(1, n_ii.size(), mxREAL);
    
    double* SumX_out = mxGetPr(plhs[1]);
    double* SumX2_out = mxGetPr(plhs[2]);
    double* n_ii_out = mxGetPr(plhs[3]);
    
    for (size_t i = 0; i < n_ii.size(); ++i)
    {
        n_ii_out[i] = n_ii[i];
        SumX_out[i] = SumX[i];
        SumX2_out[i] = SumX2[i];
    }
    
}
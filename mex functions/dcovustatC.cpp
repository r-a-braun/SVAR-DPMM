/*
* dcov.cpp - example in MATLAB External Interfaces
* 
*/

#include "mex.hpp"
#include "mexAdapter.hpp"
using namespace matlab::data;
using matlab::mex::ArgumentList;

class MexFunction : public matlab::mex::Function {
public:
    ArrayFactory factory;
    //std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
    void operator()(ArgumentList outputs, ArgumentList inputs) { 
        TypedArray<double> X = std::move(inputs[0]);
        TypedArray<double> Y = std::move(inputs[1]);  
        const double Alpha = inputs[2][0]; 
        // matlabPtr->feval(u"display",0,std::vector<Array>({ factory.createScalar(Alpha) }));  
        ArrayDimensions dim_X = X.getDimensions();
        ArrayDimensions dim_Y = Y.getDimensions();
        int n_X = dim_X[0];
        int d_X= dim_X[1];
        int n_Y = dim_Y[0]; 
        int d_Y= dim_Y[1];
        double n = dim_X[0];
        TypedArray<double>::iterator pX = X.begin();
        TypedArray<double>::iterator pY = Y.begin();
        // double* pX = X.begin(); // creates pointer for X, matrix is indexed as vector, in column order
        // double* pY = Y.begin(); // creates pointer for Y, matrix is indexed as vector, in column order
        double T1  = 0;         // creates T1  scaler for dCov
        double T2x = 0;         // creates T2x scaler for dCov, T2 = T2x*T2y
        double T2y = 0;         // creates T2y scaler for dCov, T2 = T2x*T2y
        double T3  = 0;         // creates T3  scaler for dCov
        double* pT1  = &T1;     // creates pointer for T1
        double* pT2x = &T2x;    // creates pointer for T2x
        double* pT2y = &T2y;    // creates pointer for T2y
        double* pT3  = &T3;     // creates pointer for T3
        double dX = 0;          // creates scaler for |X_k - X_l| distance dX
        double dY = 0;          // creates scaler for |Y_k - Y_l| distance dY
        double* pdX = &dX;      // creates pointer for dX
        double* pdY = &dY;      // creates pointer for dY
        std::vector<double> Xsum(n_X);      // creates vector for single sum over dX
        std::vector<double> Ysum(n_Y);      // creates vector for single sum over dX 
        std::vector<double>::iterator pXsum = Xsum.begin();     // creates pointer for Xsum
        std::vector<double>::iterator pYsum = Ysum.begin();     // creates pointer for Ysum
        
        
        int i,j,k,l = 0;                // creates indices for looping
        for (k = 1; k < n_X; k++){      // begin first loop over index k = 2:n
            for (l = 0; l < k; l++){      // begin second loop over index l < k
                
            // calculate |X_k - X_l| distance
                *pdX = 0;                   // reset dX distance to 0 via pointer
                i = 0;                      // reset column counter for X to 0
                for(i = 0; i < d_X; i++){
                    *pdX += std::pow( (pX[n_X*i+k] - pX[n_X*i+l]), 2);    // calculate sum of squared element-wise differences
                }
                *pdX = std::pow( sqrt(*pdX), Alpha);    // calculate Euclidean distance |X_k - X_l|^Alpha
                
            // calculate |Y_k - Y_l| distance
                *pdY = 0;                   // reset dY distance to 0 via pointer
                j = 0;                      // reset column counter for Y to 0
                for(j = 0; j < d_Y; j++){
                    *pdY += std::pow( (pY[n_Y*j+k] - pY[n_Y*j+l]), 2);    // calculate sum of squared element-wise differences
                }
                *pdY = std::pow( sqrt(*pdY), Alpha);    // calculate Euclidean distance |Y_k - Y_l|^Alpha
                
            // update T1, and k_th element of Xsum and Ysum
                *pT1 += (*pdX) * (*pdY);    // update T1 via pointer
                //BRisk edits 25 September 2012: changed from Xsum[k] to pXsum[k];
                //appears to be slightly faster.
                pXsum[k] += *pdX;    // update Xsum, for updating T2x and T3
                pYsum[k] += *pdY;    // update Ysum, for updating T2y and T3
                pXsum[l] += *pdX;    // update Xsum, for updating T3
                pYsum[l] += *pdY;    // update Ysum, for updating T3
                
            }    // end second loop over index l
            
            // update T2x, T2y
            *pT2x += pXsum[k] ;    // update T2x
            *pT2y += pYsum[k] ;    // update T2y
            
        }    // end first loop over index k
        
        // calculate T3
        for (k = 0; k < n_X; k++){                // begin final loop over index k
            *pT3  += pXsum[k] * pYsum[k] / 3;    // update T3 via pointer
        }                                         // end final loop over index k
        // remove extra T1 type terms
        *pT3 -= 2*(*pT1)/3;
        
        // Rf_PrintValue(Rcpp::wrap(T3));    // example print statement for debugging
        
        // update T1, T2x, T2y, T3
        *pT1  = 2 * (*pT1) * ((n) / (n-1));           // update T1 via pointer
        *pT2x = 2 * (*pT2x) / (n-1);                  // update T2x via pointer
        *pT2y = 2 * (*pT2y) / (n-1);                  // update T2y via pointer
        *pT3  = 6 * (*pT3) * ((n) / ((n-1)*(n-2)));   // update T3 via pointer
        // *pT1  = *pT1  / (n*(n-1)/2);         // update T1 via pointer
        // *pT2x = *pT2x / (n*(n-1)/2);         // update T2x via pointer
        // *pT2y = *pT2y / (n*(n-1)/2);         // update T2y via pointer
        // *pT3  = *pT3  / (n*(n-1)*(n-2)/6);   // update T3 via pointer
        
// calculate T2
        double T2 = *pT2x * *pT2y ;    // creates T2 element of dCov, T2 = T2x*T2y
        
// calculate sample distance covariance V squared
        double V2 = (*pT1 + T2 - (*pT3));

        outputs[0] = factory.createScalar(V2); 
    }
  
};
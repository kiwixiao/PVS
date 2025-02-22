/*=========================================================================
*
* Copyright Marius Staring, Stefan Klein, David Doria. 2011.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0.txt
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*=========================================================================*/
#ifndef __itkStrainEnergyVesselnessFunctor_h
#define __itkStrainEnergyVesselnessFunctor_h

#include "itkBinaryFunctorBase.h"
#include "vnl/vnl_math.h"

namespace itk
{
namespace Functor
{

/** \class StrainEnergyVesselnessFunctor
 * \brief Computes a measure of vesselness from the Hessian eigenvalues
 *        and the gradient magnitude.
 *
 * Based on the "Vesselness" measure proposed by Changyan Xiao et. al.
 *
 * \authors Changyan Xiao, Marius Staring, Denis Shamonin,
 * Johan H.C. Reiber, Jan Stolk, Berend C. Stoel
 *
 * \par Reference
 * A strain energy filter for 3D vessel enhancement with application to pulmonary CT images,
 * Medical Image Analysis, Volume 15, Issue 1, February 2011, Pages 112-124,
 * ISSN 1361-8415, DOI: 10.1016/j.media.2010.08.003.
 * http://www.sciencedirect.com/science/article/B6W6Y-5137FFY-1/2/8238fdff2ee2a26858b794913bce6546
 *
 * \sa FrangiVesselnessImageFilter
 * \ingroup IntensityImageFilters Multithreaded
 */

template< class TInput1, class TInput2, class TOutput >
class StrainEnergyVesselnessFunctor
  : public BinaryFunctorBase< TInput1, TInput2, TOutput >
{
public:
  /** Standard class typedefs. */
  typedef StrainEnergyVesselnessFunctor                   Self;
  typedef BinaryFunctorBase< TInput1, TInput2, TOutput >  Superclass;
  typedef SmartPointer< Self >                            Pointer;
  typedef SmartPointer< const Self >                      ConstPointer;

  /** New macro for creation of through a smart pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( StrainEnergyVesselnessFunctor, BinaryFunctorBase );

  /** Typedef's. */
  typedef typename NumericTraits<TOutput>::RealType RealType;
  typedef TInput2                                   EigenValueArrayType;
  typedef typename EigenValueArrayType::ValueType   EigenValueType;
  itkStaticConstMacro( Dimension, unsigned int, TInput2::Dimension );

  /** This does the real computation */
  virtual TOutput Evaluate( const TInput1 & _gradientMagnitude, const TInput2 & _eigenValues ) const
  {
    // Strain energy vesselness
    RealType SEV = NumericTraits<RealType>::Zero; // set the initial output value to 0.0

    // Static cast them first, because eigenValues could be calculated in float
    RealType eigenValues[Dimension];
    for ( unsigned int i = 0; i < Dimension; ++i )
    {
      eigenValues[ i ] = static_cast<RealType>( _eigenValues[ i ] );
    }
    const RealType gradientMagnitude = static_cast<RealType>( _gradientMagnitude ) ;

    // Calculate the maximum magnitude of eigenValues
    // i.e. the $\lambda_m$ in Eq.(15) and (27)
    // The average of eigenvalues it is also
    // 1/3 times of invariant K1 in Eq.(15)
    RealType maxEvMag = NumericTraits<RealType>::Zero;
    RealType mLamda = NumericTraits<RealType>::Zero;
    for ( unsigned int i = 0; i < Dimension; ++i )
    {
      RealType tmp1 = eigenValues[ i ];
      RealType tmp2 = vnl_math_abs( tmp1 );
      if ( maxEvMag < tmp2 )
      {
        maxEvMag = tmp2;
      }
      mLamda += tmp1;
    }
    mLamda /= Dimension;

    // Adjust contrast constraint for bright objects, see Eq.(15), Eq.(31)
    bool contrastConstraint = false;
    if( this->m_BrightObject )
    {
      contrastConstraint = ( mLamda < -maxEvMag * this->m_Alpha );
    }
    else
    {
      contrastConstraint = ( mLamda > maxEvMag * this->m_Alpha );
    }

    // Determine the contrast constraint, i.e. the "else" case in Eq.(31)
    // check: mean lambda or sum lambda? let's keep it like in paper.
    if ( contrastConstraint )
    {
      // Calculate the H2, HI2, HA2
      // i.e. |H|^2, |H^-|^2,|H^~|^2 in Eq.(A4)
      RealType H2 = NumericTraits<RealType>::Zero;
      RealType HA2 = NumericTraits<RealType>::Zero;
      for ( unsigned int i = 0; i < Dimension; ++i )
      {
        H2 += vnl_math_sqr( eigenValues[ i ] );
        HA2 += vnl_math_sqr( eigenValues[ i ] - mLamda );
      }

      const RealType HI2 = 3.0 * vnl_math_sqr( mLamda );

      // Compute the strain energy density, see Eq.(20)
      const RealType SE = vcl_sqrt( ( 1.0 - 2.0 * this->m_Nu ) * HI2
        + ( 1.0 + this->m_Nu ) * HA2 );

      // Compute the FA, see Eq.(28)
      RealType FA2 = NumericTraits<RealType>::Zero;
      FA2 = HA2 / H2;
      const RealType FA = vcl_sqrt( 3.0 * FA2 );

      // Calculate the mode(x), see Eq.(29)
      RealType tm1 = NumericTraits<RealType>::Zero;
      RealType tm2 = NumericTraits<RealType>::Zero;
      for ( unsigned int i = 0; i < Dimension; ++i )
      {
        tm1 += std::pow( eigenValues[ i ] - mLamda, static_cast<int>(Dimension) );
        tm2 += vnl_math_sqr( eigenValues[ i ] - mLamda );
      }
      tm1 /= Dimension;
      tm2 = std::pow( tm2 / 3.0, 1.5 );
      const RealType mode = vcl_sqrt( 2.0 ) * tm1 / tm2;

      // Combine FA and mode to generate the pow(V(x),k), see Eq.(30)
      if ( FA > NumericTraits<RealType>::One )
      {
        SEV = std::pow( ( mode + 1.0 ) / 2.0, this->m_Kappa );
      }
      else
      {
        RealType p1 = 0.5 * this->m_Kappa;
        SEV = std::pow( FA, p1 );
      }

      // Relative Hessian strength function, see Eq.(27)
      const RealType edgeW = vcl_exp( -this->m_Beta * gradientMagnitude / maxEvMag );

      // Integrate the above terms to form the final vessel-tuned strain energy density, see Eq.(31)
      SEV *= edgeW * SE;
    } // end if

    return static_cast<TOutput>( SEV );

  } // end Evaluate()

  /** Set parameters */
  itkSetClampMacro( Alpha, double, 0.0, 1.0 );
  itkSetClampMacro( Beta, double, 0.0, NumericTraits<double>::max() );
  itkSetClampMacro( Nu, double, -1.0, 0.5 );
  itkSetClampMacro( Kappa, double, 0.0, NumericTraits<double>::max() );
  itkSetMacro( BrightObject, bool );

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro( DimensionIs3Check,
    ( Concept::SameDimension< EigenValueArrayType::Dimension, 3 > ) );
  /** End concept checking */
#endif

protected:
  /** Constructor */
  StrainEnergyVesselnessFunctor()
  {
    this->m_Alpha = 0.0; // suggested value in the paper
    this->m_Beta = 0.3;  // suggested value in the paper
    this->m_Nu = 0.0;    // no preference for shape
    this->m_Kappa = 0.8; // suggested value in the paper
    this->m_BrightObject = true;
  };
  virtual ~StrainEnergyVesselnessFunctor(){};

private:
  StrainEnergyVesselnessFunctor(const Self &); // purposely not implemented
  void operator=(const Self &);                // purposely not implemented

  /** Member variables. */
  double  m_Alpha;
  double  m_Beta;
  double  m_Nu;
  double  m_Kappa;
  bool    m_BrightObject;

}; // end class StrainEnergyVesselnessFunctor

} // end namespace itk::Functor
} // end namespace itk

#endif

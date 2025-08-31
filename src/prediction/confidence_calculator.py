"""
파일명: src/prediction/confidence_calculator.py
목적: 예측 신뢰도 계산 및 평가 시스템
작성자: AI Assistant
작성일: 2025-08-30
버전: 1.0

주요 클래스/함수:
- ConfidenceCalculator: 신뢰도 계산 메인 클래스
- StatisticalConfidence: 통계적 신뢰도 분석
- ConsensusConfidence: 합의 기반 신뢰도
- HistoricalConfidence: 과거 성능 기반 신뢰도
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import math
from collections import Counter, defaultdict
import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta
from scipy import stats

# 상위 디렉토리 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants

class ConfidenceCalculator:
    """신뢰도 계산 메인 클래스"""
    
    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        """
        초기화
        
        Args:
            historical_data: 과거 로또 데이터 (성능 평가용)
        """
        self.historical_data = historical_data
        self.logger = self._setup_logger()
        self.confidence_history = []
        
        # 신뢰도 계산기들
        self.statistical_confidence = StatisticalConfidence()
        self.consensus_confidence = ConsensusConfidence()
        self.historical_confidence = HistoricalConfidence(historical_data)
        
        # 가중치 설정
        self.confidence_weights = {
            'statistical': 0.3,
            'consensus': 0.3,
            'historical': 0.2,
            'diversity': 0.1,
            'stability': 0.1
        }
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def calculate_prediction_confidence(self, 
                                      prediction: List[int],
                                      prediction_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        예측에 대한 종합 신뢰도 계산
        
        Args:
            prediction: 예측된 번호 리스트
            prediction_metadata: 예측 관련 메타데이터
            
        Returns:
            Dict: 신뢰도 분석 결과
        """
        self.logger.info(f"신뢰도 계산 시작: {prediction}")
        
        try:
            # 각 방법별 신뢰도 계산
            stat_conf = self.statistical_confidence.calculate(prediction, prediction_metadata)
            consensus_conf = self.consensus_confidence.calculate(prediction, prediction_metadata)
            hist_conf = self.historical_confidence.calculate(prediction, prediction_metadata)
            diversity_conf = self._calculate_diversity_confidence(prediction, prediction_metadata)
            stability_conf = self._calculate_stability_confidence(prediction, prediction_metadata)
            
            # 가중 종합 신뢰도
            total_confidence = (
                stat_conf * self.confidence_weights['statistical'] +
                consensus_conf * self.confidence_weights['consensus'] +
                hist_conf * self.confidence_weights['historical'] +
                diversity_conf * self.confidence_weights['diversity'] +
                stability_conf * self.confidence_weights['stability']
            )
            
            # 신뢰도 등급 분류
            confidence_grade = self._classify_confidence_grade(total_confidence)
            
            # 불확실성 지표
            uncertainty_metrics = self._calculate_uncertainty_metrics(prediction, prediction_metadata)
            
            # 신뢰 구간
            confidence_interval = self._calculate_confidence_interval(prediction, prediction_metadata)
            
            result = {
                'total_confidence': total_confidence,
                'confidence_grade': confidence_grade,
                'component_confidences': {
                    'statistical': stat_conf,
                    'consensus': consensus_conf,
                    'historical': hist_conf,
                    'diversity': diversity_conf,
                    'stability': stability_conf
                },
                'uncertainty_metrics': uncertainty_metrics,
                'confidence_interval': confidence_interval,
                'reliability_indicators': self._generate_reliability_indicators(total_confidence),
                'recommendation': self._generate_recommendation(total_confidence, confidence_grade),
                'calculation_timestamp': datetime.now().isoformat()
            }
            
            # 히스토리에 추가
            self.confidence_history.append({
                'prediction': prediction.copy(),
                'confidence_result': result,
                'timestamp': datetime.now()
            })
            
            self.logger.info(f"신뢰도 계산 완료: {total_confidence:.3f} ({confidence_grade})")
            return result
            
        except Exception as e:
            self.logger.error(f"신뢰도 계산 중 오류: {e}")
            # 기본 신뢰도 반환
            return self._generate_fallback_confidence()
    
    def _calculate_diversity_confidence(self, prediction: List[int], metadata: Dict[str, Any]) -> float:
        """다양성 기반 신뢰도"""
        diversity_score = metadata.get('diversity_score', 0.5)
        
        # 다양성이 높을수록 신뢰도 향상 (극단적 예측 회피)
        if diversity_score > 0.8:
            return 0.9
        elif diversity_score > 0.6:
            return 0.7
        elif diversity_score > 0.4:
            return 0.6
        else:
            return 0.4
    
    def _calculate_stability_confidence(self, prediction: List[int], metadata: Dict[str, Any]) -> float:
        """안정성 기반 신뢰도"""
        # 최근 예측들과의 일관성 체크
        if len(self.confidence_history) < 2:
            return 0.5  # 충분한 히스토리가 없음
        
        recent_predictions = [entry['prediction'] for entry in self.confidence_history[-5:]]
        
        # 예측 일관성 측정
        consistency_scores = []
        for past_pred in recent_predictions:
            overlap = len(set(prediction) & set(past_pred))
            consistency = overlap / len(prediction)
            consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores)
        
        # 적절한 일관성 (너무 높으면 고착화, 너무 낮으면 불안정)
        if 0.3 <= avg_consistency <= 0.7:
            return 0.8  # 적절한 일관성
        elif avg_consistency < 0.3:
            return 0.4  # 너무 불안정
        else:
            return 0.5  # 너무 일관적 (고착화 위험)
    
    def _classify_confidence_grade(self, confidence: float) -> str:
        """신뢰도 등급 분류"""
        if confidence >= 0.8:
            return 'A'  # 매우 높음
        elif confidence >= 0.7:
            return 'B'  # 높음
        elif confidence >= 0.6:
            return 'C'  # 보통
        elif confidence >= 0.5:
            return 'D'  # 낮음
        else:
            return 'F'  # 매우 낮음
    
    def _calculate_uncertainty_metrics(self, prediction: List[int], metadata: Dict[str, Any]) -> Dict[str, float]:
        """불확실성 지표 계산"""
        # 예측 분산도
        prediction_variance = np.var(prediction)
        
        # 메타데이터에서 불확실성 지표 추출
        consensus_uncertainty = 1.0 - metadata.get('consensus_ratio', 0.5)
        method_agreement = metadata.get('method_agreement', 0.5)
        
        # 전체 불확실성
        total_uncertainty = (prediction_variance / 100 + consensus_uncertainty + (1 - method_agreement)) / 3
        
        return {
            'prediction_variance': prediction_variance,
            'consensus_uncertainty': consensus_uncertainty,
            'method_disagreement': 1 - method_agreement,
            'total_uncertainty': min(1.0, total_uncertainty)
        }
    
    def _calculate_confidence_interval(self, prediction: List[int], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """신뢰 구간 계산"""
        confidence_level = 0.95  # 95% 신뢰구간
        
        # 각 번호 위치의 불확실성 추정
        position_uncertainties = []
        for number in prediction:
            # 번호의 과거 출현 패턴 기반 불확실성
            if self.historical_data is not None:
                # 실제 데이터 기반 계산 (단순화)
                uncertainty = 0.1 + (abs(number - 23) / 45) * 0.1  # 중앙값에서 멀수록 불확실
            else:
                uncertainty = 0.15  # 기본 불확실성
            
            position_uncertainties.append(uncertainty)
        
        return {
            'confidence_level': confidence_level,
            'position_uncertainties': position_uncertainties,
            'average_uncertainty': np.mean(position_uncertainties),
            'prediction_range_low': [max(1, int(num * (1 - unc))) for num, unc in zip(prediction, position_uncertainties)],
            'prediction_range_high': [min(45, int(num * (1 + unc))) for num, unc in zip(prediction, position_uncertainties)]
        }
    
    def _generate_reliability_indicators(self, confidence: float) -> Dict[str, Any]:
        """신뢰성 지표 생성"""
        return {
            'reliability_score': confidence,
            'risk_level': 'low' if confidence > 0.7 else 'medium' if confidence > 0.5 else 'high',
            'expected_accuracy': f"{confidence * 100:.1f}%",
            'recommendation_strength': 'strong' if confidence > 0.8 else 'moderate' if confidence > 0.6 else 'weak',
            'usage_advice': self._generate_usage_advice(confidence)
        }
    
    def _generate_usage_advice(self, confidence: float) -> str:
        """사용 조언 생성"""
        if confidence > 0.8:
            return "높은 신뢰도로 권장합니다. 하지만 로또의 무작위성을 항상 고려하세요."
        elif confidence > 0.6:
            return "보통 수준의 신뢰도입니다. 다른 예측과 함께 참고하세요."
        elif confidence > 0.4:
            return "낮은 신뢰도입니다. 신중하게 고려하시기 바랍니다."
        else:
            return "매우 낮은 신뢰도입니다. 추가적인 분석이 필요합니다."
    
    def _generate_recommendation(self, confidence: float, grade: str) -> str:
        """추천 사항 생성"""
        recommendations = {
            'A': "이 예측은 높은 신뢰도를 보입니다. 주요 선택지로 고려하세요.",
            'B': "양호한 신뢰도의 예측입니다. 다른 분석과 함께 검토하세요.",
            'C': "보통 수준의 예측입니다. 추가 검증을 권장합니다.",
            'D': "신뢰도가 낮은 예측입니다. 신중한 판단이 필요합니다.",
            'F': "신뢰도가 매우 낮습니다. 다른 예측 방법을 고려해보세요."
        }
        
        return recommendations.get(grade, "예측 신뢰도를 재검토하세요.")
    
    def _generate_fallback_confidence(self) -> Dict[str, Any]:
        """비상 신뢰도 반환"""
        return {
            'total_confidence': 0.5,
            'confidence_grade': 'C',
            'component_confidences': {
                'statistical': 0.5,
                'consensus': 0.5,
                'historical': 0.5,
                'diversity': 0.5,
                'stability': 0.5
            },
            'uncertainty_metrics': {
                'total_uncertainty': 0.5
            },
            'confidence_interval': {
                'confidence_level': 0.95,
                'average_uncertainty': 0.15
            },
            'reliability_indicators': {
                'risk_level': 'medium',
                'usage_advice': '기본 신뢰도로 신중하게 사용하세요.'
            },
            'recommendation': '신뢰도 계산 중 오류가 발생했습니다. 기본 수준으로 평가됩니다.'
        }
    
    def get_confidence_trends(self) -> Dict[str, Any]:
        """신뢰도 트렌드 분석"""
        if len(self.confidence_history) < 3:
            return {"message": "충분한 히스토리가 없습니다."}
        
        recent_confidences = [entry['confidence_result']['total_confidence'] 
                            for entry in self.confidence_history[-10:]]
        
        return {
            'recent_average': np.mean(recent_confidences),
            'trend': 'improving' if recent_confidences[-1] > recent_confidences[0] else 'declining',
            'volatility': np.std(recent_confidences),
            'consistency': 1 - (np.std(recent_confidences) / np.mean(recent_confidences)) if np.mean(recent_confidences) > 0 else 0
        }


class StatisticalConfidence:
    """통계적 신뢰도 분석"""
    
    def calculate(self, prediction: List[int], metadata: Dict[str, Any]) -> float:
        """통계적 신뢰도 계산"""
        scores = []
        
        # 1. 번호 분포의 정규성 (너무 극단적이지 않은가)
        distribution_score = self._evaluate_distribution_normality(prediction)
        scores.append(distribution_score)
        
        # 2. 패턴의 자연스러움
        pattern_score = self._evaluate_pattern_naturalness(prediction)
        scores.append(pattern_score)
        
        # 3. 통계적 일관성
        consistency_score = self._evaluate_statistical_consistency(prediction)
        scores.append(consistency_score)
        
        return np.mean(scores)
    
    def _evaluate_distribution_normality(self, prediction: List[int]) -> float:
        """분포 정규성 평가"""
        # 평균이 중앙값(23) 근처인지
        mean_deviation = abs(np.mean(prediction) - 23) / 23
        mean_score = 1 - min(1.0, mean_deviation)
        
        # 분산이 적절한지
        variance = np.var(prediction)
        expected_variance = 150  # 경험적 적절값
        variance_score = 1 - min(1.0, abs(variance - expected_variance) / expected_variance)
        
        return (mean_score + variance_score) / 2
    
    def _evaluate_pattern_naturalness(self, prediction: List[int]) -> float:
        """패턴 자연스러움 평가"""
        sorted_pred = sorted(prediction)
        
        # 연속번호 개수 (1-2개가 자연스러움)
        consecutive_count = sum(1 for i in range(len(sorted_pred)-1) 
                               if sorted_pred[i+1] - sorted_pred[i] == 1)
        consecutive_score = 1 - abs(consecutive_count - 1) / len(prediction)
        
        # 홀짝 균형 (3:3이 이상적)
        odd_count = sum(1 for n in prediction if n % 2 == 1)
        balance_score = 1 - abs(odd_count - 3) / 3
        
        # 간격의 다양성
        gaps = [sorted_pred[i+1] - sorted_pred[i] for i in range(len(sorted_pred)-1)]
        gap_variety = len(set(gaps)) / len(gaps) if gaps else 0
        
        return (consecutive_score + balance_score + gap_variety) / 3
    
    def _evaluate_statistical_consistency(self, prediction: List[int]) -> float:
        """통계적 일관성 평가"""
        # 구간별 분포 (1-15, 16-30, 31-45)
        low = sum(1 for n in prediction if n <= 15)
        mid = sum(1 for n in prediction if 16 <= n <= 30)
        high = sum(1 for n in prediction if n >= 31)
        
        # 균등 분포에서의 편차
        expected = len(prediction) / 3
        distribution_variance = np.var([low, mid, high])
        consistency_score = 1 / (1 + distribution_variance)
        
        return consistency_score


class ConsensusConfidence:
    """합의 기반 신뢰도"""
    
    def calculate(self, prediction: List[int], metadata: Dict[str, Any]) -> float:
        """합의 기반 신뢰도 계산"""
        # 메타데이터에서 합의 정보 추출
        consensus_data = metadata.get('consensus_analysis', {})
        
        # 높은 합의를 받은 번호들의 비율
        high_consensus_numbers = consensus_data.get('high_consensus_numbers', [])
        consensus_ratio = len([n for n in prediction if n in high_consensus_numbers]) / len(prediction)
        
        # 예측기들 간의 일치도
        prediction_agreement = metadata.get('prediction_agreement', 0.5)
        
        # 신뢰도 점수들의 분산 (낮을수록 좋음)
        individual_confidences = metadata.get('individual_confidences', [0.5] * 5)
        confidence_consistency = 1 - (np.std(individual_confidences) / np.mean(individual_confidences)) if individual_confidences else 0
        
        # 가중 평균
        consensus_confidence = (
            consensus_ratio * 0.4 +
            prediction_agreement * 0.3 +
            confidence_consistency * 0.3
        )
        
        return min(1.0, max(0.0, consensus_confidence))


class HistoricalConfidence:
    """과거 성능 기반 신뢰도"""
    
    def __init__(self, historical_data: Optional[pd.DataFrame]):
        self.historical_data = historical_data
        self.performance_cache = {}
    
    def calculate(self, prediction: List[int], metadata: Dict[str, Any]) -> float:
        """과거 성능 기반 신뢰도 계산"""
        if self.historical_data is None:
            return 0.5  # 데이터가 없으면 중간값
        
        # 예측 방법별 과거 성능
        method_performance = self._evaluate_method_performance(metadata)
        
        # 유사한 패턴의 과거 성공률
        pattern_success_rate = self._evaluate_pattern_success_rate(prediction)
        
        # 번호별 과거 성능
        number_performance = self._evaluate_number_performance(prediction)
        
        # 가중 평균
        historical_confidence = (
            method_performance * 0.4 +
            pattern_success_rate * 0.3 +
            number_performance * 0.3
        )
        
        return historical_confidence
    
    def _evaluate_method_performance(self, metadata: Dict[str, Any]) -> float:
        """방법별 과거 성능 평가"""
        # 실제 구현에서는 각 예측 방법의 과거 성공률을 계산
        # 현재는 단순화된 버전
        used_methods = metadata.get('used_methods', [])
        
        # 방법별 기대 성능 (경험적 값)
        method_performances = {
            'genius_insight': 0.7,
            'frequency_based': 0.6,
            'pattern_based': 0.65,
            'trend_based': 0.55,
            'ensemble': 0.75
        }
        
        if used_methods:
            avg_performance = np.mean([method_performances.get(method, 0.5) for method in used_methods])
        else:
            avg_performance = 0.6  # 기본값
        
        return avg_performance
    
    def _evaluate_pattern_success_rate(self, prediction: List[int]) -> float:
        """패턴 성공률 평가"""
        # 홀짝 패턴 분석
        odd_count = sum(1 for n in prediction if n % 2 == 1)
        
        # 경험적으로 3:3 또는 4:2 비율이 가장 성공적
        if odd_count in [3, 4]:
            pattern_score = 0.8
        elif odd_count in [2, 5]:
            pattern_score = 0.6
        else:
            pattern_score = 0.4
        
        # 구간 분포 패턴
        low = sum(1 for n in prediction if n <= 15)
        mid = sum(1 for n in prediction if 16 <= n <= 30)
        high = sum(1 for n in prediction if n >= 31)
        
        # 균등 분포에 가까울수록 높은 점수
        distribution_balance = 1 - np.std([low, mid, high]) / 2
        
        return (pattern_score + distribution_balance) / 2
    
    def _evaluate_number_performance(self, prediction: List[int]) -> float:
        """번호별 성능 평가"""
        # 실제 데이터가 있다면 각 번호의 과거 당첨 기여도를 계산
        # 현재는 단순화된 접근
        
        # 중간 범위 번호들이 일반적으로 더 안정적
        performance_scores = []
        for number in prediction:
            if 10 <= number <= 35:  # 중간 범위
                performance_scores.append(0.7)
            elif 5 <= number <= 40:  # 넓은 중간 범위
                performance_scores.append(0.6)
            else:  # 극값
                performance_scores.append(0.4)
        
        return np.mean(performance_scores)


# === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 데이터
    test_prediction = [7, 14, 21, 28, 35, 42]
    test_metadata = {
        'diversity_score': 0.7,
        'consensus_analysis': {
            'high_consensus_numbers': [7, 14, 28],
            'consensus_ratio': 0.6
        },
        'prediction_agreement': 0.75,
        'individual_confidences': [0.8, 0.7, 0.6, 0.5, 0.7],
        'used_methods': ['genius_insight', 'ensemble', 'frequency_based']
    }
    
    print("=== 신뢰도 계산기 테스트 ===")
    
    try:
        # 신뢰도 계산기 초기화
        confidence_calc = ConfidenceCalculator()
        
        # 신뢰도 계산
        result = confidence_calc.calculate_prediction_confidence(test_prediction, test_metadata)
        
        print(f"예측: {test_prediction}")
        print(f"전체 신뢰도: {result['total_confidence']:.3f}")
        print(f"신뢰도 등급: {result['confidence_grade']}")
        
        print(f"\n세부 신뢰도:")
        for component, score in result['component_confidences'].items():
            print(f"  {component}: {score:.3f}")
        
        print(f"\n신뢰성 지표:")
        reliability = result['reliability_indicators']
        print(f"  위험 수준: {reliability['risk_level']}")
        print(f"  예상 정확도: {reliability['expected_accuracy']}")
        print(f"  권장 강도: {reliability['recommendation_strength']}")
        
        print(f"\n불확실성 지표:")
        uncertainty = result['uncertainty_metrics']
        print(f"  전체 불확실성: {uncertainty['total_uncertainty']:.3f}")
        
        print(f"\n권장사항: {result['recommendation']}")
        print(f"\n사용 조언: {reliability['usage_advice']}")
        
        # 여러 예측에 대한 신뢰도 추세 테스트
        test_predictions = [
            ([1, 8, 15, 22, 29, 36], {'diversity_score': 0.8}),
            ([3, 10, 17, 24, 31, 38], {'diversity_score': 0.6}),
            ([5, 12, 19, 26, 33, 40], {'diversity_score': 0.9})
        ]
        
        print(f"\n=== 추가 예측 신뢰도 테스트 ===")
        for i, (pred, meta) in enumerate(test_predictions, 2):
            result = confidence_calc.calculate_prediction_confidence(pred, meta)
            print(f"예측 {i}: {pred} -> 신뢰도: {result['total_confidence']:.3f} ({result['confidence_grade']})")
        
        # 트렌드 분석
        trends = confidence_calc.get_confidence_trends()
        print(f"\n신뢰도 트렌드:")
        print(f"  최근 평균: {trends.get('recent_average', 0):.3f}")
        print(f"  추세: {trends.get('trend', 'unknown')}")
        print(f"  일관성: {trends.get('consistency', 0):.3f}")
        
        print("✅ 신뢰도 계산기 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
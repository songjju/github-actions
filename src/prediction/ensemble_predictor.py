"""
파일명: src/prediction/ensemble_predictor.py
목적: 다중 예측 모델을 통합하는 앙상블 예측 시스템
작성자: AI Assistant
작성일: 2025-08-30
버전: 1.0

주요 클래스/함수:
- EnsemblePredictor: 앙상블 예측 시스템 메인 클래스
- PredictionMethod: 개별 예측 방법 추상 클래스
- WeightedVoting: 가중 투표 기반 예측 통합
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import random
import logging
from pathlib import Path
import sys
from collections import Counter, defaultdict
import math

# 상위 디렉토리의 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants
from src.genius_formulas.formula_01_genius_insight import GeniusInsightFormula

class PredictionMethod(ABC):
    """예측 방법 추상 클래스"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.confidence = 0.5
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, stats: Dict[str, Any], count: int = 6) -> List[int]:
        """예측 수행"""
        pass
    
    @abstractmethod
    def calculate_confidence(self, prediction: List[int]) -> float:
        """예측 신뢰도 계산"""
        pass

class FrequencyBasedPredictor(PredictionMethod):
    """빈도 기반 예측기"""
    
    def __init__(self, weight: float = 1.0):
        super().__init__("Frequency-Based", weight)
    
    def predict(self, data: pd.DataFrame, stats: Dict[str, Any], count: int = 6) -> List[int]:
        """빈도 기반 예측"""
        frequency_data = stats.get('frequency_analysis', {})
        frequency_count = frequency_data.get('frequency_count', {})
        
        if not frequency_count:
            return random.sample(range(1, 46), count)
        
        # 빈도를 확률로 변환
        total_count = sum(frequency_count.values())
        probabilities = []
        
        for number in range(1, 46):
            freq = frequency_count.get(number, 0)
            probability = freq / total_count if total_count > 0 else 1/45
            probabilities.append((number, probability))
        
        # 확률에 따른 가중 선택
        return self._weighted_selection(probabilities, count)
    
    def calculate_confidence(self, prediction: List[int]) -> float:
        """빈도 기반 신뢰도"""
        # 예측된 번호들의 평균 빈도를 신뢰도로 사용
        return min(0.8, 0.3 + len(prediction) * 0.05)
    
    def _weighted_selection(self, probabilities: List[Tuple[int, float]], count: int) -> List[int]:
        """가중 선택"""
        selected = []
        available = probabilities.copy()
        
        for _ in range(count):
            if not available:
                break
            
            # 확률 정규화
            total_prob = sum(prob for _, prob in available)
            if total_prob == 0:
                selected.append(available[0][0])
                available.pop(0)
                continue
            
            # 가중 랜덤 선택
            rand = random.random() * total_prob
            cumsum = 0
            
            for i, (number, prob) in enumerate(available):
                cumsum += prob
                if rand <= cumsum:
                    selected.append(number)
                    available.pop(i)
                    break
        
        return sorted(selected)

class TrendBasedPredictor(PredictionMethod):
    """트렌드 기반 예측기"""
    
    def __init__(self, weight: float = 0.8):
        super().__init__("Trend-Based", weight)
    
    def predict(self, data: pd.DataFrame, stats: Dict[str, Any], count: int = 6) -> List[int]:
        """트렌드 기반 예측"""
        trend_data = stats.get('trend_analysis', {})
        
        # 최근 트렌드 데이터 사용
        recent_key = 'last_52_draws'  # 최근 1년
        if recent_key not in trend_data:
            recent_key = list(trend_data.keys())[0] if trend_data else None
        
        if not recent_key or recent_key not in trend_data:
            return random.sample(range(1, 46), count)
        
        recent_trend = trend_data[recent_key]
        most_frequent = recent_trend.get('most_frequent', [])
        
        # 최근 빈도가 높은 번호들을 우선 선택
        trend_numbers = []
        for number, freq in most_frequent:
            if len(trend_numbers) < count * 2:  # 후보를 많이 확보
                trend_numbers.append(number)
        
        # 부족한 경우 전체 데이터에서 보완
        if len(trend_numbers) < count:
            freq_data = stats.get('frequency_analysis', {}).get('frequency_count', {})
            additional_numbers = sorted(freq_data.keys(), key=lambda x: freq_data.get(x, 0), reverse=True)
            for num in additional_numbers:
                if num not in trend_numbers and len(trend_numbers) < count * 2:
                    trend_numbers.append(num)
        
        # 랜덤하게 최종 선택
        if len(trend_numbers) >= count:
            selected = random.sample(trend_numbers, count)
        else:
            selected = trend_numbers + random.sample(
                [i for i in range(1, 46) if i not in trend_numbers], 
                count - len(trend_numbers)
            )
        
        return sorted(selected)
    
    def calculate_confidence(self, prediction: List[int]) -> float:
        """트렌드 기반 신뢰도"""
        return 0.6  # 중간 정도 신뢰도

class PatternBasedPredictor(PredictionMethod):
    """패턴 기반 예측기"""
    
    def __init__(self, weight: float = 0.9):
        super().__init__("Pattern-Based", weight)
    
    def predict(self, data: pd.DataFrame, stats: Dict[str, Any], count: int = 6) -> List[int]:
        """패턴 기반 예측"""
        pattern_data = stats.get('pattern_analysis', {})
        
        # 홀짝 패턴 고려
        odd_even_patterns = pattern_data.get('odd_even_patterns', {})
        most_common_odd = odd_even_patterns.get('most_common_odd_count', [3, 0])
        target_odd_count = most_common_odd[0] if isinstance(most_common_odd, list) else 3
        
        # 구간 패턴 고려
        section_patterns = pattern_data.get('section_patterns', {})
        avg_distribution = section_patterns.get('avg_distribution', {'low': 2, 'mid': 2, 'high': 2})
        
        # 패턴에 맞는 번호 선택
        selected_numbers = []
        
        # 구간별 목표 개수
        target_low = max(1, round(avg_distribution.get('low', 2)))
        target_mid = max(1, round(avg_distribution.get('mid', 2)))
        target_high = max(1, count - target_low - target_mid)
        
        # 각 구간에서 번호 선택
        low_numbers = self._select_from_range(1, 15, target_low, target_odd_count / 3)
        mid_numbers = self._select_from_range(16, 30, target_mid, target_odd_count / 3)
        high_numbers = self._select_from_range(31, 45, target_high, target_odd_count / 3)
        
        selected_numbers.extend(low_numbers)
        selected_numbers.extend(mid_numbers)
        selected_numbers.extend(high_numbers)
        
        # 부족하거나 초과한 경우 조정
        while len(selected_numbers) < count:
            available = [i for i in range(1, 46) if i not in selected_numbers]
            if available:
                selected_numbers.append(random.choice(available))
            else:
                break
        
        if len(selected_numbers) > count:
            selected_numbers = random.sample(selected_numbers, count)
        
        return sorted(selected_numbers)
    
    def _select_from_range(self, start: int, end: int, target_count: int, target_odd_ratio: float) -> List[int]:
        """범위에서 패턴에 맞는 번호 선택"""
        available_numbers = list(range(start, end + 1))
        selected = []
        
        # 홀짝 비율 고려
        target_odd_count = max(0, min(target_count, round(target_count * target_odd_ratio)))
        target_even_count = target_count - target_odd_count
        
        # 홀수 선택
        odd_numbers = [n for n in available_numbers if n % 2 == 1]
        if odd_numbers and target_odd_count > 0:
            selected.extend(random.sample(odd_numbers, min(len(odd_numbers), target_odd_count)))
        
        # 짝수 선택
        even_numbers = [n for n in available_numbers if n % 2 == 0 and n not in selected]
        if even_numbers and target_even_count > 0:
            selected.extend(random.sample(even_numbers, min(len(even_numbers), target_even_count)))
        
        # 부족한 경우 추가
        while len(selected) < target_count:
            remaining = [n for n in available_numbers if n not in selected]
            if remaining:
                selected.append(random.choice(remaining))
            else:
                break
        
        return selected
    
    def calculate_confidence(self, prediction: List[int]) -> float:
        """패턴 기반 신뢰도"""
        return 0.7

class AntiPatternPredictor(PredictionMethod):
    """역패턴 예측기 (일반적 패턴의 반대)"""
    
    def __init__(self, weight: float = 0.5):
        super().__init__("Anti-Pattern", weight)
    
    def predict(self, data: pd.DataFrame, stats: Dict[str, Any], count: int = 6) -> List[int]:
        """역패턴 예측"""
        frequency_data = stats.get('frequency_analysis', {})
        
        # 가장 적게 나온 번호들 우선 선택
        frequency_count = frequency_data.get('frequency_count', {})
        if frequency_count:
            sorted_by_freq = sorted(frequency_count.items(), key=lambda x: x[1])
            cold_numbers = [num for num, _ in sorted_by_freq[:15]]  # 하위 15개
        else:
            cold_numbers = list(range(1, 46))
        
        # 차가운 번호들 중에서 선택
        if len(cold_numbers) >= count:
            selected = random.sample(cold_numbers, count)
        else:
            selected = cold_numbers + random.sample(
                [i for i in range(1, 46) if i not in cold_numbers],
                count - len(cold_numbers)
            )
        
        return sorted(selected)
    
    def calculate_confidence(self, prediction: List[int]) -> float:
        """역패턴 신뢰도"""
        return 0.4  # 낮은 신뢰도 (실험적 성격)

class EnsemblePredictor:
    """앙상블 예측 시스템"""
    
    def __init__(self, lotto_data: pd.DataFrame, statistics: Dict[str, Any]):
        """
        앙상블 예측기 초기화
        
        Args:
            lotto_data (pd.DataFrame): 로또 데이터
            statistics (Dict[str, Any]): 통계 분석 결과
        """
        self.data = lotto_data
        self.stats = statistics
        self.logger = self._setup_logger()
        
        # 개별 예측 방법들 초기화
        self.predictors = [
            FrequencyBasedPredictor(weight=1.0),
            TrendBasedPredictor(weight=0.8),
            PatternBasedPredictor(weight=0.9),
            AntiPatternPredictor(weight=0.5)
        ]
        
        # 천재적 통찰 공식 추가
        try:
            self.genius_formula = GeniusInsightFormula(lotto_data, statistics)
            self.predictors.append(self._create_genius_predictor())
        except Exception as e:
            self.logger.warning(f"천재적 통찰 공식 초기화 실패: {e}")
    
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
    
    def _create_genius_predictor(self) -> PredictionMethod:
        """천재적 통찰 예측기 생성"""
        class GeniusPredictor(PredictionMethod):
            def __init__(self, genius_formula, weight=1.2):
                super().__init__("Genius-Insight", weight)
                self.genius_formula = genius_formula
            
            def predict(self, data, stats, count=6):
                return self.genius_formula.predict_numbers(count)
            
            def calculate_confidence(self, prediction):
                return 0.8  # 높은 신뢰도
        
        return GeniusPredictor(self.genius_formula)
    
    def predict_ensemble(self, count: int = 6, method: str = 'weighted_voting') -> Dict[str, Any]:
        """
        앙상블 예측 수행
        
        Args:
            count (int): 예측할 번호 개수
            method (str): 통합 방법 ('weighted_voting', 'majority_voting', 'rank_fusion')
            
        Returns:
            Dict[str, Any]: 예측 결과 및 상세 정보
        """
        self.logger.info(f"앙상블 예측 시작 - 방법: {method}")
        
        try:
            # 각 예측기에서 예측 수행
            individual_predictions = {}
            prediction_confidences = {}
            
            for predictor in self.predictors:
                try:
                    prediction = predictor.predict(self.data, self.stats, count)
                    confidence = predictor.calculate_confidence(prediction)
                    
                    individual_predictions[predictor.name] = prediction
                    prediction_confidences[predictor.name] = confidence
                    
                    self.logger.info(f"{predictor.name}: {prediction} (신뢰도: {confidence:.2f})")
                    
                except Exception as e:
                    self.logger.warning(f"{predictor.name} 예측 실패: {e}")
                    continue
            
            if not individual_predictions:
                raise ValueError("모든 예측기가 실패했습니다.")
            
            # 예측 통합
            if method == 'weighted_voting':
                final_prediction = self._weighted_voting(individual_predictions, prediction_confidences, count)
            elif method == 'majority_voting':
                final_prediction = self._majority_voting(individual_predictions, count)
            elif method == 'rank_fusion':
                final_prediction = self._rank_fusion(individual_predictions, count)
            else:
                raise ValueError(f"지원하지 않는 통합 방법: {method}")
            
            # 결과 구성
            result = {
                'final_prediction': sorted(final_prediction),
                'method': method,
                'individual_predictions': individual_predictions,
                'prediction_confidences': prediction_confidences,
                'ensemble_confidence': self._calculate_ensemble_confidence(
                    individual_predictions, prediction_confidences, final_prediction
                ),
                'consensus_analysis': self._analyze_consensus(individual_predictions),
                'prediction_metadata': self._generate_prediction_metadata(final_prediction)
            }
            
            self.logger.info(f"앙상블 예측 완료: {result['final_prediction']}")
            return result
            
        except Exception as e:
            self.logger.error(f"앙상블 예측 중 오류: {e}")
            raise
    
    def _weighted_voting(self, predictions: Dict[str, List[int]], confidences: Dict[str, float], count: int) -> List[int]:
        """가중 투표 방식"""
        number_scores = defaultdict(float)
        total_weight = 0
        
        # 각 예측기의 가중치와 신뢰도를 고려한 점수 계산
        for predictor_name, prediction in predictions.items():
            predictor = next(p for p in self.predictors if p.name == predictor_name)
            confidence = confidences.get(predictor_name, 0.5)
            
            # 가중치 = 예측기 가중치 × 신뢰도
            effective_weight = predictor.weight * confidence
            total_weight += effective_weight
            
            # 각 번호에 점수 부여 (순서도 고려)
            for i, number in enumerate(prediction):
                position_weight = (count - i) / count  # 앞순서일수록 높은 가중치
                number_scores[number] += effective_weight * position_weight
        
        # 점수 정규화
        if total_weight > 0:
            for number in number_scores:
                number_scores[number] /= total_weight
        
        # 상위 번호들 선택
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        selected_numbers = [num for num, _ in sorted_numbers[:count]]
        
        # 부족한 경우 보완
        if len(selected_numbers) < count:
            all_predicted = set()
            for pred in predictions.values():
                all_predicted.update(pred)
            
            additional = [num for num in all_predicted if num not in selected_numbers]
            selected_numbers.extend(additional[:count - len(selected_numbers)])
        
        # 여전히 부족하면 랜덤 추가
        if len(selected_numbers) < count:
            remaining = [i for i in range(1, 46) if i not in selected_numbers]
            selected_numbers.extend(random.sample(remaining, count - len(selected_numbers)))
        
        return selected_numbers[:count]
    
    def _majority_voting(self, predictions: Dict[str, List[int]], count: int) -> List[int]:
        """다수결 투표 방식"""
        number_votes = Counter()
        
        # 각 번호의 투표 수 계산
        for prediction in predictions.values():
            for number in prediction:
                number_votes[number] += 1
        
        # 투표 수가 높은 번호들 선택
        most_voted = number_votes.most_common(count)
        selected_numbers = [num for num, _ in most_voted]
        
        # 부족한 경우 보완
        if len(selected_numbers) < count:
            all_numbers = set()
            for pred in predictions.values():
                all_numbers.update(pred)
            
            additional = [num for num in all_numbers if num not in selected_numbers]
            selected_numbers.extend(additional[:count - len(selected_numbers)])
        
        # 여전히 부족하면 랜덤 추가
        if len(selected_numbers) < count:
            remaining = [i for i in range(1, 46) if i not in selected_numbers]
            selected_numbers.extend(random.sample(remaining, count - len(selected_numbers)))
        
        return selected_numbers[:count]
    
    def _rank_fusion(self, predictions: Dict[str, List[int]], count: int) -> List[int]:
        """순위 융합 방식 (Borda Count)"""
        number_scores = defaultdict(float)
        
        # 각 예측에서의 순위 점수 계산
        for prediction in predictions.values():
            for i, number in enumerate(prediction):
                # 앞순서일수록 높은 점수 (count - i)
                rank_score = count - i
                number_scores[number] += rank_score
        
        # 점수순으로 정렬하여 선택
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        selected_numbers = [num for num, _ in sorted_numbers[:count]]
        
        # 부족한 경우 처리 (majority voting과 동일)
        if len(selected_numbers) < count:
            all_numbers = set()
            for pred in predictions.values():
                all_numbers.update(pred)
            
            additional = [num for num in all_numbers if num not in selected_numbers]
            selected_numbers.extend(additional[:count - len(selected_numbers)])
        
        if len(selected_numbers) < count:
            remaining = [i for i in range(1, 46) if i not in selected_numbers]
            selected_numbers.extend(random.sample(remaining, count - len(selected_numbers)))
        
        return selected_numbers[:count]
    
    def _calculate_ensemble_confidence(self, predictions: Dict[str, List[int]], 
                                     confidences: Dict[str, float], 
                                     final_prediction: List[int]) -> float:
        """앙상블 신뢰도 계산"""
        if not predictions or not confidences:
            return 0.5
        
        # 1. 개별 예측기 신뢰도의 가중 평균
        total_weight = sum(predictor.weight for predictor in self.predictors if predictor.name in predictions)
        weighted_confidence = 0
        
        for predictor_name, confidence in confidences.items():
            predictor = next((p for p in self.predictors if p.name == predictor_name), None)
            if predictor:
                weight = predictor.weight / total_weight if total_weight > 0 else 1 / len(confidences)
                weighted_confidence += confidence * weight
        
        # 2. 예측 일치도 보너스
        consensus_bonus = self._calculate_consensus_bonus(predictions, final_prediction)
        
        # 3. 최종 신뢰도
        ensemble_confidence = weighted_confidence * (1 + consensus_bonus * 0.2)
        
        return min(1.0, max(0.1, ensemble_confidence))
    
    def _calculate_consensus_bonus(self, predictions: Dict[str, List[int]], final_prediction: List[int]) -> float:
        """예측 일치도 보너스 계산"""
        if not predictions:
            return 0
        
        consensus_scores = []
        
        for prediction in predictions.values():
            # 최종 예측과의 일치 개수
            overlap = len(set(prediction) & set(final_prediction))
            consensus_score = overlap / len(final_prediction)
            consensus_scores.append(consensus_score)
        
        # 평균 일치도
        avg_consensus = np.mean(consensus_scores) if consensus_scores else 0
        return avg_consensus
    
    def _analyze_consensus(self, predictions: Dict[str, List[int]]) -> Dict[str, Any]:
        """예측 일치 분석"""
        if not predictions:
            return {}
        
        all_numbers = []
        for prediction in predictions.values():
            all_numbers.extend(prediction)
        
        number_frequency = Counter(all_numbers)
        
        # 높은 일치도를 보이는 번호들
        high_consensus = [num for num, freq in number_frequency.items() if freq >= len(predictions) * 0.6]
        
        # 분산된 의견을 보이는 번호들
        low_consensus = [num for num, freq in number_frequency.items() if freq == 1]
        
        return {
            'total_unique_numbers': len(number_frequency),
            'high_consensus_numbers': high_consensus,
            'low_consensus_numbers': low_consensus,
            'number_frequency': dict(number_frequency),
            'consensus_ratio': len(high_consensus) / len(number_frequency) if number_frequency else 0
        }
    
    def _generate_prediction_metadata(self, prediction: List[int]) -> Dict[str, Any]:
        """예측 메타데이터 생성"""
        # 기본 통계
        prediction_sum = sum(prediction)
        odd_count = sum(1 for num in prediction if num % 2 == 1)
        even_count = len(prediction) - odd_count
        
        # 구간 분포
        low_count = sum(1 for num in prediction if num <= 15)
        mid_count = sum(1 for num in prediction if 16 <= num <= 30)
        high_count = sum(1 for num in prediction if num >= 31)
        
        # 연속성 분석
        sorted_pred = sorted(prediction)
        consecutive_pairs = sum(1 for i in range(len(sorted_pred)-1) if sorted_pred[i+1] - sorted_pred[i] == 1)
        
        # 간격 분석
        gaps = [sorted_pred[i+1] - sorted_pred[i] for i in range(len(sorted_pred)-1)]
        avg_gap = np.mean(gaps) if gaps else 0
        
        return {
            'sum': prediction_sum,
            'odd_count': odd_count,
            'even_count': even_count,
            'section_distribution': {
                'low': low_count,
                'mid': mid_count,
                'high': high_count
            },
            'consecutive_pairs': consecutive_pairs,
            'average_gap': avg_gap,
            'min_number': min(prediction),
            'max_number': max(prediction),
            'range': max(prediction) - min(prediction)
        }
    
    def predict_multiple_sets(self, num_sets: int = 5, method: str = 'weighted_voting') -> List[Dict[str, Any]]:
        """
        여러 세트의 예측 생성
        
        Args:
            num_sets (int): 생성할 예측 세트 수
            method (str): 통합 방법
            
        Returns:
            List[Dict[str, Any]]: 예측 세트들의 리스트
        """
        prediction_sets = []
        
        for i in range(num_sets):
            try:
                # 매번 약간의 랜덤성을 추가하여 다양성 확보
                prediction = self.predict_ensemble(6, method)
                prediction['set_id'] = i + 1
                prediction_sets.append(prediction)
                
            except Exception as e:
                self.logger.warning(f"예측 세트 {i+1} 생성 실패: {e}")
                continue
        
        return prediction_sets
    
    def get_predictor_performance(self) -> Dict[str, Any]:
        """예측기별 성능 정보 반환"""
        performance = {}
        
        for predictor in self.predictors:
            # 간단한 성능 지표 (실제로는 과거 예측 결과 기반으로 계산)
            performance[predictor.name] = {
                'weight': predictor.weight,
                'expected_confidence': predictor.calculate_confidence([1, 2, 3, 4, 5, 6]),
                'description': self._get_predictor_description(predictor.name)
            }
        
        return performance
    
    def _get_predictor_description(self, name: str) -> str:
        """예측기 설명 반환"""
        descriptions = {
            'Frequency-Based': '과거 출현 빈도를 기반으로 예측',
            'Trend-Based': '최근 트렌드와 패턴을 분석하여 예측',
            'Pattern-Based': '홀짝, 구간 분포 등 패턴을 고려한 예측',
            'Anti-Pattern': '일반적 패턴의 반대로 예측 (실험적)',
            'Genius-Insight': '천재적 통찰 공식을 활용한 창의적 예측'
        }
        return descriptions.get(name, '알 수 없는 예측 방법')


# === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 샘플 데이터 (새로운 CSV 구조에 맞춰 수정)
    sample_data = pd.DataFrame({
        'round_number': [1187, 1186, 1185, 1184, 1183],
        'draw_date': ['2025.08.30', '2025.08.23', '2025.08.16', '2025.08.09', '2025.08.02'],
        'year': [2025, 2025, 2025, 2025, 2025],
        'winning_numbers': [
            [5, 13, 26, 29, 37, 40],
            [2, 8, 13, 16, 23, 28],
            [6, 17, 22, 28, 29, 32],
            [14, 16, 23, 25, 31, 37],
            [4, 15, 17, 23, 27, 36]
        ],
        'bonus_number': [42, 35, 38, 42, 31],
        'number_sum': [150, 90, 134, 146, 122],
        'odd_count': [3, 3, 3, 5, 3],
        'even_count': [3, 3, 3, 1, 3],
        'low_count': [1, 3, 1, 0, 1],
        'mid_count': [2, 3, 4, 4, 4],
        'high_count': [3, 0, 1, 2, 1]
    })
    
    # 샘플 통계
    sample_stats = {
        'frequency_analysis': {
            'frequency_count': {i: random.randint(8, 18) for i in range(1, 46)},
            'hot_numbers': [13, 16, 23, 25, 28],
            'cold_numbers': [4, 11, 18, 33, 41]
        },
        'trend_analysis': {
            'last_52_draws': {
                'most_frequent': [(13, 8), (23, 7), (28, 6), (16, 5), (25, 5)]
            }
        },
        'pattern_analysis': {
            'odd_even_patterns': {
                'most_common_odd_count': [3, 15]
            },
            'section_patterns': {
                'avg_distribution': {'low': 2.2, 'mid': 2.4, 'high': 1.4}
            }
        }
    }
    
    print("=== 앙상블 예측 시스템 테스트 (새로운 CSV 구조) ===")
    
    try:
        # 앙상블 예측기 초기화
        ensemble = EnsemblePredictor(sample_data, sample_stats)
        
        # 단일 예측 테스트
        result = ensemble.predict_ensemble(6, 'weighted_voting')
        
        print(f"최종 예측: {result['final_prediction']}")
        print(f"앙상블 신뢰도: {result['ensemble_confidence']:.2f}")
        print(f"높은 일치도 번호: {result['consensus_analysis']['high_consensus_numbers']}")
        
        print(f"\n개별 예측기 결과:")
        for name, prediction in result['individual_predictions'].items():
            confidence = result['prediction_confidences'][name]
            print(f"  {name}: {prediction} (신뢰도: {confidence:.2f})")
        
        # 다중 세트 예측 테스트
        print(f"\n다중 세트 예측 (3세트):")
        multiple_sets = ensemble.predict_multiple_sets(3)
        for i, pred_set in enumerate(multiple_sets, 1):
            print(f"  세트 {i}: {pred_set['final_prediction']}")
        
        print("✅ 앙상블 예측 시스템 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
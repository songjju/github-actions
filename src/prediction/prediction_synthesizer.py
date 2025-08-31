"""
파일명: src/prediction/prediction_synthesizer.py
목적: 예측 결과 합성 및 최종 번호 선택 로직
작성자: AI Assistant
작성일: 2025-08-30
버전: 1.0

주요 클래스/함수:
- PredictionSynthesizer: 예측 결과 통합 및 합성
- AlternativeGenerator: 대안 예측 생성
- ResultOptimizer: 결과 최적화
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter, defaultdict
import random
import math
import logging
from pathlib import Path
import sys
from datetime import datetime, timedelta

# 상위 디렉토리 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants, validate_lotto_numbers

class PredictionSynthesizer:
    """예측 결과 합성 및 통합 클래스"""
    
    def __init__(self):
        """초기화"""
        self.logger = self._setup_logger()
        self.synthesis_history = []
        self.optimization_strategies = [
            'balanced_selection',
            'entropy_maximization', 
            'pattern_diversity',
            'statistical_harmony'
        ]
        
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
    
    def synthesize_predictions(self, 
                             ensemble_results: List[Dict[str, Any]], 
                             target_count: int = 6,
                             strategy: str = 'comprehensive') -> Dict[str, Any]:
        """
        여러 예측 결과를 합성하여 최적의 예측 생성
        
        Args:
            ensemble_results (List[Dict]): 앙상블 예측 결과들
            target_count (int): 목표 번호 개수
            strategy (str): 합성 전략
            
        Returns:
            Dict[str, Any]: 합성된 최종 예측
        """
        self.logger.info(f"예측 합성 시작 - 전략: {strategy}")
        
        try:
            if not ensemble_results:
                raise ValueError("합성할 예측 결과가 없습니다.")
            
            # 전략에 따른 합성 수행
            if strategy == 'comprehensive':
                final_prediction = self._comprehensive_synthesis(ensemble_results, target_count)
            elif strategy == 'confidence_weighted':
                final_prediction = self._confidence_weighted_synthesis(ensemble_results, target_count)
            elif strategy == 'consensus_based':
                final_prediction = self._consensus_based_synthesis(ensemble_results, target_count)
            elif strategy == 'diversity_optimized':
                final_prediction = self._diversity_optimized_synthesis(ensemble_results, target_count)
            else:
                self.logger.warning(f"알 수 없는 전략: {strategy}, 기본 전략 사용")
                final_prediction = self._comprehensive_synthesis(ensemble_results, target_count)
            
            # 합성 결과 검증
            if not validate_lotto_numbers(final_prediction):
                self.logger.warning("합성 결과가 유효하지 않음, 대안 생성")
                final_prediction = self._generate_fallback_prediction(target_count)
            
            # 합성 메타데이터 생성
            synthesis_metadata = self._generate_synthesis_metadata(
                ensemble_results, final_prediction, strategy
            )
            
            result = {
                'final_prediction': sorted(final_prediction),
                'synthesis_strategy': strategy,
                'source_predictions': len(ensemble_results),
                'synthesis_metadata': synthesis_metadata,
                'quality_metrics': self._calculate_quality_metrics(final_prediction),
                'alternatives': self._generate_alternatives(ensemble_results, final_prediction, 3)
            }
            
            # 히스토리에 추가
            self.synthesis_history.append({
                'timestamp': datetime.now().isoformat(),
                'result': result,
                'input_count': len(ensemble_results)
            })
            
            self.logger.info(f"예측 합성 완료: {result['final_prediction']}")
            return result
            
        except Exception as e:
            self.logger.error(f"예측 합성 중 오류: {e}")
            raise
    
    def _comprehensive_synthesis(self, ensemble_results: List[Dict], target_count: int) -> List[int]:
        """포괄적 합성 전략"""
        # 모든 예측에서 번호별 점수 집계
        number_scores = defaultdict(float)
        total_weight = 0
        
        for result in ensemble_results:
            prediction = result.get('final_prediction', [])
            confidence = result.get('ensemble_confidence', 0.5)
            
            # 가중치 계산
            weight = confidence * len(prediction) / target_count
            total_weight += weight
            
            # 번호별 점수 부여 (위치도 고려)
            for i, number in enumerate(prediction):
                position_bonus = (target_count - i) / target_count  # 앞순서 보너스
                number_scores[number] += weight * (1.0 + position_bonus * 0.2)
        
        # 점수 정규화
        if total_weight > 0:
            for number in number_scores:
                number_scores[number] /= total_weight
        
        # 상위 번호들 선택 + 다양성 고려
        selected_numbers = self._select_with_diversity(number_scores, target_count)
        
        return selected_numbers
    
    def _confidence_weighted_synthesis(self, ensemble_results: List[Dict], target_count: int) -> List[int]:
        """신뢰도 가중 합성"""
        weighted_votes = Counter()
        
        for result in ensemble_results:
            prediction = result.get('final_prediction', [])
            confidence = result.get('ensemble_confidence', 0.5)
            
            # 신뢰도를 가중치로 사용
            for number in prediction:
                weighted_votes[number] += confidence
        
        # 가중 투표 결과 기반 선택
        most_voted = weighted_votes.most_common(target_count * 2)  # 여유분 확보
        candidate_numbers = [num for num, _ in most_voted]
        
        # 다양성을 고려한 최종 선택
        final_numbers = self._ensure_diversity(candidate_numbers[:target_count], target_count)
        
        return final_numbers
    
    def _consensus_based_synthesis(self, ensemble_results: List[Dict], target_count: int) -> List[int]:
        """합의 기반 합성"""
        all_predictions = []
        for result in ensemble_results:
            prediction = result.get('final_prediction', [])
            all_predictions.append(prediction)
        
        # 출현 빈도 계산
        number_frequency = Counter()
        for prediction in all_predictions:
            for number in prediction:
                number_frequency[number] += 1
        
        # 높은 합의를 받은 번호들 우선 선택
        high_consensus = []
        medium_consensus = []
        low_consensus = []
        
        threshold_high = len(all_predictions) * 0.6
        threshold_medium = len(all_predictions) * 0.3
        
        for number, freq in number_frequency.items():
            if freq >= threshold_high:
                high_consensus.append(number)
            elif freq >= threshold_medium:
                medium_consensus.append(number)
            else:
                low_consensus.append(number)
        
        # 계층적 선택
        selected = []
        selected.extend(high_consensus[:target_count])
        
        if len(selected) < target_count:
            remaining = target_count - len(selected)
            selected.extend(medium_consensus[:remaining])
        
        if len(selected) < target_count:
            remaining = target_count - len(selected)
            selected.extend(random.sample(low_consensus, min(remaining, len(low_consensus))))
        
        # 부족한 경우 보완
        if len(selected) < target_count:
            all_numbers = set(range(1, 46))
            available = list(all_numbers - set(selected))
            selected.extend(random.sample(available, target_count - len(selected)))
        
        return selected[:target_count]
    
    def _diversity_optimized_synthesis(self, ensemble_results: List[Dict], target_count: int) -> List[int]:
        """다양성 최적화 합성"""
        # 모든 후보 번호 수집
        all_candidates = set()
        for result in ensemble_results:
            prediction = result.get('final_prediction', [])
            all_candidates.update(prediction)
        
        # 다양성 지표를 최대화하는 조합 탐색
        best_combination = None
        best_diversity_score = -1
        
        # 상위 후보들로부터 조합 생성
        candidates_list = list(all_candidates)
        
        # 여러 조합을 시도하여 최적 다양성 찾기
        for _ in range(min(100, len(candidates_list))):  # 최대 100회 시도
            if len(candidates_list) >= target_count:
                candidate_combination = random.sample(candidates_list, target_count)
                diversity_score = self._calculate_diversity_score(candidate_combination)
                
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_combination = candidate_combination
        
        if best_combination is None:
            # 실패 시 랜덤 선택
            best_combination = random.sample(list(range(1, 46)), target_count)
        
        return best_combination
    
    def _select_with_diversity(self, number_scores: Dict[int, float], target_count: int) -> List[int]:
        """다양성을 고려한 번호 선택"""
        sorted_numbers = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected = []
        considered = []
        
        # 상위 후보들 수집 (target_count * 2)
        for number, score in sorted_numbers[:target_count * 2]:
            considered.append(number)
        
        # 다양성을 고려한 선택
        while len(selected) < target_count and considered:
            if not selected:
                # 첫 번째는 최고 점수
                selected.append(considered.pop(0))
            else:
                # 나머지는 다양성 고려
                best_candidate = None
                best_diversity = -1
                
                for candidate in considered:
                    test_selection = selected + [candidate]
                    diversity = self._calculate_diversity_score(test_selection)
                    if diversity > best_diversity:
                        best_diversity = diversity
                        best_candidate = candidate
                
                if best_candidate is not None:
                    selected.append(best_candidate)
                    considered.remove(best_candidate)
                else:
                    # 다양성 계산 실패시 첫 번째 선택
                    selected.append(considered.pop(0))
        
        # 부족한 경우 보완
        if len(selected) < target_count:
            remaining_numbers = [i for i in range(1, 46) if i not in selected]
            selected.extend(random.sample(remaining_numbers, target_count - len(selected)))
        
        return selected[:target_count]
    
    def _calculate_diversity_score(self, numbers: List[int]) -> float:
        """다양성 점수 계산"""
        if len(numbers) < 2:
            return 0
        
        scores = []
        
        # 1. 홀짝 균형
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        odd_balance = 1 - abs(odd_count - len(numbers)/2) / (len(numbers)/2)
        scores.append(odd_balance)
        
        # 2. 구간 분포
        low = sum(1 for n in numbers if n <= 15)
        mid = sum(1 for n in numbers if 16 <= n <= 30)
        high = sum(1 for n in numbers if n >= 31)
        
        expected_per_section = len(numbers) / 3
        section_variance = np.var([low, mid, high])
        section_balance = 1 / (1 + section_variance)
        scores.append(section_balance)
        
        # 3. 번호 간격 다양성
        sorted_numbers = sorted(numbers)
        gaps = [sorted_numbers[i+1] - sorted_numbers[i] for i in range(len(sorted_numbers)-1)]
        gap_diversity = 1 - (np.std(gaps) / np.mean(gaps)) if gaps else 0
        scores.append(max(0, gap_diversity))
        
        # 4. 연속성 패널티
        consecutive_count = sum(1 for i in range(len(sorted_numbers)-1) 
                               if sorted_numbers[i+1] - sorted_numbers[i] == 1)
        consecutive_penalty = 1 - (consecutive_count / len(numbers))
        scores.append(consecutive_penalty)
        
        return np.mean(scores)
    
    def _ensure_diversity(self, numbers: List[int], target_count: int) -> List[int]:
        """다양성 보장 후처리"""
        if len(numbers) != target_count:
            if len(numbers) > target_count:
                numbers = numbers[:target_count]
            else:
                remaining = [i for i in range(1, 46) if i not in numbers]
                numbers.extend(random.sample(remaining, target_count - len(numbers)))
        
        # 극단적 패턴 보정
        sorted_nums = sorted(numbers)
        
        # 모두 홀수이거나 모두 짝수인 경우 보정
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        if odd_count == 0 or odd_count == len(numbers):
            # 하나씩 교체하여 균형 맞추기
            target_odd = len(numbers) // 2
            if odd_count < target_odd:
                # 홀수가 부족한 경우
                evens_to_replace = [n for n in numbers if n % 2 == 0]
                available_odds = [n for n in range(1, 46, 2) if n not in numbers]
                if evens_to_replace and available_odds:
                    numbers[numbers.index(evens_to_replace[0])] = available_odds[0]
            elif odd_count > target_odd:
                # 홀수가 과다한 경우
                odds_to_replace = [n for n in numbers if n % 2 == 1]
                available_evens = [n for n in range(2, 46, 2) if n not in numbers]
                if odds_to_replace and available_evens:
                    numbers[numbers.index(odds_to_replace[0])] = available_evens[0]
        
        return numbers
    
    def _generate_synthesis_metadata(self, ensemble_results: List[Dict], 
                                   final_prediction: List[int], strategy: str) -> Dict[str, Any]:
        """합성 메타데이터 생성"""
        # 입력 예측들과의 유사도 계산
        similarities = []
        for result in ensemble_results:
            prediction = result.get('final_prediction', [])
            similarity = len(set(final_prediction) & set(prediction)) / len(final_prediction)
            similarities.append(similarity)
        
        # 번호별 출현 빈도
        number_frequency = Counter()
        for result in ensemble_results:
            prediction = result.get('final_prediction', [])
            for number in prediction:
                number_frequency[number] += 1
        
        return {
            'strategy_used': strategy,
            'input_predictions': len(ensemble_results),
            'average_similarity': np.mean(similarities) if similarities else 0,
            'max_similarity': max(similarities) if similarities else 0,
            'min_similarity': min(similarities) if similarities else 0,
            'number_consensus': {
                num: freq for num, freq in number_frequency.items() 
                if num in final_prediction
            },
            'diversity_score': self._calculate_diversity_score(final_prediction),
            'synthesis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_quality_metrics(self, prediction: List[int]) -> Dict[str, float]:
        """예측 품질 지표 계산"""
        return {
            'diversity_score': self._calculate_diversity_score(prediction),
            'balance_score': self._calculate_balance_score(prediction),
            'pattern_score': self._calculate_pattern_score(prediction),
            'overall_quality': self._calculate_overall_quality(prediction)
        }
    
    def _calculate_balance_score(self, numbers: List[int]) -> float:
        """균형 점수 계산"""
        if not numbers:
            return 0
        
        # 홀짝 균형
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        odd_balance = 1 - abs(odd_count - len(numbers)/2) / (len(numbers)/2)
        
        # 크기 균형 (작은수/큰수)
        avg_number = np.mean(numbers)
        expected_avg = 23  # 1-45의 중간값
        size_balance = 1 - abs(avg_number - expected_avg) / expected_avg
        
        return (odd_balance + size_balance) / 2
    
    def _calculate_pattern_score(self, numbers: List[int]) -> float:
        """패턴 점수 계산"""
        sorted_nums = sorted(numbers)
        
        # 연속번호 적절성 (너무 많거나 적으면 부자연스러움)
        consecutive = sum(1 for i in range(len(sorted_nums)-1) 
                         if sorted_nums[i+1] - sorted_nums[i] == 1)
        consecutive_score = 1 - abs(consecutive - 1) / len(numbers)  # 1개 정도가 적절
        
        # 간격 분산 (너무 규칙적이면 부자연스러움)
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
        if gaps:
            gap_variance = np.var(gaps)
            gap_score = min(1.0, gap_variance / 10)  # 적절한 분산
        else:
            gap_score = 0
        
        return (consecutive_score + gap_score) / 2
    
    def _calculate_overall_quality(self, numbers: List[int]) -> float:
        """전체 품질 점수 계산"""
        diversity = self._calculate_diversity_score(numbers)
        balance = self._calculate_balance_score(numbers)
        pattern = self._calculate_pattern_score(numbers)
        
        # 가중 평균
        return diversity * 0.4 + balance * 0.3 + pattern * 0.3
    
    def _generate_alternatives(self, ensemble_results: List[Dict], 
                             main_prediction: List[int], count: int) -> List[Dict]:
        """대안 예측들 생성"""
        alternatives = []
        
        # 다른 합성 전략들로 대안 생성
        strategies = ['confidence_weighted', 'consensus_based', 'diversity_optimized']
        
        for strategy in strategies[:count]:
            try:
                if strategy == 'confidence_weighted':
                    alt_prediction = self._confidence_weighted_synthesis(ensemble_results, 6)
                elif strategy == 'consensus_based':
                    alt_prediction = self._consensus_based_synthesis(ensemble_results, 6)
                else:  # diversity_optimized
                    alt_prediction = self._diversity_optimized_synthesis(ensemble_results, 6)
                
                # 메인 예측과 다른 경우만 추가
                if set(alt_prediction) != set(main_prediction):
                    alternatives.append({
                        'prediction': sorted(alt_prediction),
                        'strategy': strategy,
                        'similarity_to_main': len(set(alt_prediction) & set(main_prediction)) / 6,
                        'quality_score': self._calculate_overall_quality(alt_prediction)
                    })
                    
            except Exception as e:
                self.logger.warning(f"대안 생성 중 오류 ({strategy}): {e}")
                continue
        
        # 품질 점수로 정렬
        alternatives.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return alternatives[:count]
    
    def _generate_fallback_prediction(self, target_count: int) -> List[int]:
        """비상 예측 생성"""
        self.logger.warning("비상 예측 생성")
        return sorted(random.sample(range(1, 46), target_count))
    
    def get_synthesis_history(self) -> List[Dict]:
        """합성 히스토리 반환"""
        return self.synthesis_history.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        if not self.synthesis_history:
            return {"message": "합성 히스토리가 없습니다."}
        
        # 품질 점수 통계
        quality_scores = []
        diversity_scores = []
        
        for entry in self.synthesis_history:
            result = entry['result']
            quality_metrics = result.get('quality_metrics', {})
            
            if 'overall_quality' in quality_metrics:
                quality_scores.append(quality_metrics['overall_quality'])
            if 'diversity_score' in quality_metrics:
                diversity_scores.append(quality_metrics['diversity_score'])
        
        return {
            'total_syntheses': len(self.synthesis_history),
            'average_quality': np.mean(quality_scores) if quality_scores else 0,
            'average_diversity': np.mean(diversity_scores) if diversity_scores else 0,
            'quality_trend': 'improving' if len(quality_scores) > 1 and quality_scores[-1] > quality_scores[0] else 'stable'
        }


class AlternativeGenerator:
    """대안 예측 생성기"""
    
    def __init__(self):
        self.generation_methods = [
            'random_variation',
            'number_substitution', 
            'pattern_modification',
            'statistical_rebalancing'
        ]
    
    def generate_alternatives(self, base_prediction: List[int], 
                            count: int = 3, 
                            variation_degree: str = 'medium') -> List[Dict]:
        """
        기본 예측을 바탕으로 대안들 생성
        
        Args:
            base_prediction: 기본 예측
            count: 생성할 대안 수
            variation_degree: 변화 정도 ('low', 'medium', 'high')
            
        Returns:
            List[Dict]: 대안 예측들
        """
        alternatives = []
        
        variation_rates = {
            'low': 0.2,      # 20% 변경
            'medium': 0.4,   # 40% 변경  
            'high': 0.6      # 60% 변경
        }
        
        change_rate = variation_rates.get(variation_degree, 0.4)
        change_count = max(1, int(len(base_prediction) * change_rate))
        
        for i in range(count):
            method = self.generation_methods[i % len(self.generation_methods)]
            
            if method == 'random_variation':
                alternative = self._random_variation(base_prediction, change_count)
            elif method == 'number_substitution':
                alternative = self._number_substitution(base_prediction, change_count)
            elif method == 'pattern_modification':
                alternative = self._pattern_modification(base_prediction, change_count)
            else:  # statistical_rebalancing
                alternative = self._statistical_rebalancing(base_prediction, change_count)
            
            alternatives.append({
                'prediction': sorted(alternative),
                'method': method,
                'changes_made': change_count,
                'similarity': len(set(alternative) & set(base_prediction)) / len(base_prediction)
            })
        
        return alternatives
    
    def _random_variation(self, base: List[int], change_count: int) -> List[int]:
        """랜덤 변형"""
        result = base.copy()
        available = [i for i in range(1, 46) if i not in base]
        
        # 랜덤하게 번호 교체
        for _ in range(min(change_count, len(available))):
            if result and available:
                old_idx = random.randint(0, len(result) - 1)
                new_number = random.choice(available)
                
                available.append(result[old_idx])  # 제거된 번호를 사용가능 목록에 추가
                available.remove(new_number)       # 새 번호를 사용가능 목록에서 제거
                result[old_idx] = new_number
        
        return result
    
    def _number_substitution(self, base: List[int], change_count: int) -> List[int]:
        """유사 번호로 치환"""
        result = base.copy()
        
        for _ in range(change_count):
            if result:
                # 변경할 번호 선택
                old_number = random.choice(result)
                old_idx = result.index(old_number)
                
                # 유사한 번호로 치환 (±5 범위)
                candidates = []
                for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    candidate = old_number + delta
                    if 1 <= candidate <= 45 and candidate not in result:
                        candidates.append(candidate)
                
                if candidates:
                    result[old_idx] = random.choice(candidates)
        
        return result
    
    def _pattern_modification(self, base: List[int], change_count: int) -> List[int]:
        """패턴 기반 수정"""
        result = base.copy()
        
        # 홀짝 비율 조정
        odd_count = sum(1 for n in result if n % 2 == 1)
        target_odd = random.choice([2, 3, 4])  # 목표 홀수 개수
        
        if odd_count != target_odd:
            # 홀짝 비율 조정을 위한 교체
            if odd_count < target_odd:
                # 홀수 부족 - 짝수를 홀수로 교체
                evens = [n for n in result if n % 2 == 0]
                available_odds = [n for n in range(1, 46, 2) if n not in result]
                
                for _ in range(min(target_odd - odd_count, len(evens), len(available_odds))):
                    if evens and available_odds:
                        old_even = random.choice(evens)
                        new_odd = random.choice(available_odds)
                        
                        result[result.index(old_even)] = new_odd
                        evens.remove(old_even)
                        available_odds.remove(new_odd)
            else:
                # 홀수 과다 - 홀수를 짝수로 교체
                odds = [n for n in result if n % 2 == 1]
                available_evens = [n for n in range(2, 46, 2) if n not in result]
                
                for _ in range(min(odd_count - target_odd, len(odds), len(available_evens))):
                    if odds and available_evens:
                        old_odd = random.choice(odds)
                        new_even = random.choice(available_evens)
                        
                        result[result.index(old_odd)] = new_even
                        odds.remove(old_odd)
                        available_evens.remove(new_even)
        
        return result
    
    def _statistical_rebalancing(self, base: List[int], change_count: int) -> List[int]:
        """통계적 재균형"""
        result = base.copy()
        
        # 구간별 분포 재조정
        low_count = sum(1 for n in result if n <= 15)
        mid_count = sum(1 for n in result if 16 <= n <= 30)
        high_count = sum(1 for n in result if n >= 31)
        
        # 이상적 분포 (각 구간 2개씩)
        target_distribution = [2, 2, 2]
        current_distribution = [low_count, mid_count, high_count]
        
        # 가장 불균형한 구간 조정
        for _ in range(min(change_count, 3)):
            # 과다 구간과 부족 구간 찾기
            over_idx = -1
            under_idx = -1
            max_over = 0
            max_under = 0
            
            for i in range(3):
                diff = current_distribution[i] - target_distribution[i]
                if diff > max_over:
                    max_over = diff
                    over_idx = i
                if diff < max_under:
                    max_under = diff
                    under_idx = i
            
            if over_idx != -1 and under_idx != -1 and over_idx != under_idx:
                # 과다 구간에서 부족 구간으로 이동
                over_ranges = [(1, 15), (16, 30), (31, 45)]
                over_range = over_ranges[over_idx]
                under_range = over_ranges[under_idx]
                
                # 이동할 번호 선택
                candidates_to_move = [n for n in result if over_range[0] <= n <= over_range[1]]
                available_targets = [n for n in range(under_range[0], under_range[1] + 1) if n not in result]
                
                if candidates_to_move and available_targets:
                    old_number = random.choice(candidates_to_move)
                    new_number = random.choice(available_targets)
                    
                    result[result.index(old_number)] = new_number
                    current_distribution[over_idx] -= 1
                    current_distribution[under_idx] += 1
        
        return result


# === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 앙상블 결과
    test_ensemble_results = [
        {
            'final_prediction': [1, 7, 14, 21, 28, 35],
            'ensemble_confidence': 0.8,
            'method': 'genius_insight'
        },
        {
            'final_prediction': [3, 9, 15, 22, 29, 36],
            'ensemble_confidence': 0.7,
            'method': 'frequency_based'
        },
        {
            'final_prediction': [5, 12, 19, 26, 33, 40],
            'ensemble_confidence': 0.6,
            'method': 'pattern_based'
        },
        {
            'final_prediction': [2, 8, 16, 23, 30, 37],
            'ensemble_confidence': 0.5,
            'method': 'trend_based'
        }
    ]
    
    print("=== 예측 합성기 테스트 ===")
    
    try:
        # 합성기 초기화
        synthesizer = PredictionSynthesizer()
        
        # 다양한 전략으로 합성 테스트
        strategies = ['comprehensive', 'confidence_weighted', 'consensus_based', 'diversity_optimized']
        
        for strategy in strategies:
            print(f"\n🔄 {strategy} 전략 테스트:")
            result = synthesizer.synthesize_predictions(test_ensemble_results, 6, strategy)
            
            print(f"  최종 예측: {result['final_prediction']}")
            print(f"  다양성 점수: {result['quality_metrics']['diversity_score']:.3f}")
            print(f"  전체 품질: {result['quality_metrics']['overall_quality']:.3f}")
            
            if result['alternatives']:
                print(f"  대안 1: {result['alternatives'][0]['prediction']}")
        
        # 대안 생성기 테스트
        print(f"\n🎲 대안 생성기 테스트:")
        alt_generator = AlternativeGenerator()
        
        base_prediction = [1, 7, 14, 21, 28, 35]
        alternatives = alt_generator.generate_alternatives(base_prediction, 3, 'medium')
        
        print(f"기본 예측: {base_prediction}")
        for i, alt in enumerate(alternatives, 1):
            print(f"  대안 {i} ({alt['method']}): {alt['prediction']} (유사도: {alt['similarity']:.2f})")
        
        # 성능 통계
        print(f"\n📊 성능 통계:")
        stats = synthesizer.get_performance_stats()
        print(f"  총 합성 횟수: {stats['total_syntheses']}")
        print(f"  평균 품질: {stats['average_quality']:.3f}")
        print(f"  평균 다양성: {stats['average_diversity']:.3f}")
        
        print("✅ 예측 합성기 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
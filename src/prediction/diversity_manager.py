"""
파일명: src/prediction/diversity_manager.py
목적: 예측 다양성 보장 시스템 - 동일 예측 방지 및 번호 고착화 방지
작성자: AI Assistant
작성일: 2025-08-30
버전: 1.0

핵심 기능:
- 완전 동일 예측 방지
- 특정 번호 과도 사용 방지  
- 예측 히스토리 관리
- 동적 다양성 조절

클래스:
- PredictionHistory: 예측 이력 관리
- DiversityGuaranteedPredictor: 다양성 보장 예측기
- AdaptiveDiversityController: 적응형 다양성 제어기
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter, defaultdict, deque
import random
import math
import logging
from pathlib import Path
import sys
import json
from datetime import datetime

# 상위 디렉토리 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, LottoConstants

class PredictionHistory:
    """예측 히스토리 관리 클래스"""
    
    def __init__(self, max_history: int = 1000):
        """
        예측 히스토리 초기화
        
        Args:
            max_history (int): 최대 히스토리 보관 개수
        """
        self.max_history = max_history
        self.prediction_history = deque(maxlen=max_history)  # 전체 예측 이력
        self.recent_predictions = deque(maxlen=50)  # 최근 예측 (중복 체크용)
        self.number_usage_count = {i: 0 for i in range(1, 46)}  # 번호별 사용 횟수
        self.prediction_count = 0  # 총 예측 횟수
        self.diversity_metrics = {}  # 다양성 지표
        
    def add_prediction(self, numbers: List[int], metadata: Optional[Dict] = None):
        """
        새로운 예측을 히스토리에 추가
        
        Args:
            numbers (List[int]): 예측된 번호 리스트
            metadata (Optional[Dict]): 예측 메타데이터
        """
        if not self._validate_prediction(numbers):
            raise ValueError(f"유효하지 않은 예측: {numbers}")
        
        # 예측 정보 구성
        prediction_info = {
            'numbers': sorted(numbers.copy()),
            'timestamp': datetime.now().isoformat(),
            'prediction_id': self.prediction_count + 1,
            'metadata': metadata or {}
        }
        
        # 히스토리에 추가
        self.prediction_history.append(prediction_info)
        self.recent_predictions.append(sorted(numbers.copy()))
        
        # 번호별 사용 횟수 업데이트
        for num in numbers:
            self.number_usage_count[num] += 1
        
        self.prediction_count += 1
        
        # 다양성 지표 업데이트
        self._update_diversity_metrics()
    
    def is_duplicate_prediction(self, numbers: List[int]) -> bool:
        """
        완전 동일한 예측인지 확인
        
        Args:
            numbers (List[int]): 확인할 번호 리스트
            
        Returns:
            bool: 중복 여부
        """
        sorted_numbers = sorted(numbers)
        
        for past_prediction in self.recent_predictions:
            if past_prediction == sorted_numbers:
                return True
        
        return False
    
    def get_overused_numbers(self, threshold_ratio: float = 0.3) -> List[int]:
        """
        과도하게 사용된 번호들 반환
        
        Args:
            threshold_ratio (float): 임계치 비율
            
        Returns:
            List[int]: 과도 사용 번호 리스트
        """
        if self.prediction_count == 0:
            return []
        
        # 이론적 평균 사용 횟수
        expected_usage = self.prediction_count * 6 / 45
        threshold = expected_usage * (1 + threshold_ratio)
        
        overused = []
        for number, count in self.number_usage_count.items():
            if count > threshold:
                overused.append(number)
        
        return sorted(overused)
    
    def get_underused_numbers(self, threshold_ratio: float = 0.3) -> List[int]:
        """
        과소 사용된 번호들 반환
        
        Args:
            threshold_ratio (float): 임계치 비율
            
        Returns:
            List[int]: 과소 사용 번호 리스트
        """
        if self.prediction_count == 0:
            return list(range(1, 46))
        
        expected_usage = self.prediction_count * 6 / 45
        threshold = expected_usage * (1 - threshold_ratio)
        
        underused = []
        for number, count in self.number_usage_count.items():
            if count < threshold:
                underused.append(number)
        
        return sorted(underused)
    
    def calculate_diversity_health(self) -> float:
        """
        현재 다양성 건강도 측정 (0.0~1.0)
        
        Returns:
            float: 다양성 건강도 점수
        """
        if len(self.recent_predictions) < 3:
            return 1.0  # 충분한 데이터가 없으면 최대값
        
        scores = []
        
        # 1. 최근 예측의 고유성
        recent_uniqueness = self._measure_recent_uniqueness()
        scores.append(recent_uniqueness)
        
        # 2. 번호 분포의 엔트로피
        distribution_entropy = self._measure_distribution_entropy()
        scores.append(distribution_entropy)
        
        # 3. 패턴 다양성
        pattern_diversity = self._measure_pattern_diversity()
        scores.append(pattern_diversity)
        
        return sum(scores) / len(scores)
    
    def _measure_recent_uniqueness(self) -> float:
        """최근 예측들의 고유성 측정"""
        if len(self.recent_predictions) < 10:
            return 1.0
        
        recent_10 = list(self.recent_predictions)[-10:]
        unique_predictions = []
        
        for pred in recent_10:
            pred_tuple = tuple(pred)
            if pred_tuple not in [tuple(up) for up in unique_predictions]:
                unique_predictions.append(pred)
        
        return len(unique_predictions) / len(recent_10)
    
    def _measure_distribution_entropy(self) -> float:
        """번호 분포의 엔트로피 측정"""
        if self.prediction_count == 0:
            return 1.0
        
        usage_counts = list(self.number_usage_count.values())
        total_usage = sum(usage_counts)
        
        if total_usage == 0:
            return 1.0
        
        # 엔트로피 계산
        entropy = 0
        for count in usage_counts:
            if count > 0:
                probability = count / total_usage
                entropy -= probability * math.log2(probability)
        
        # 최대 엔트로피로 정규화 (45개 번호가 균등 분포일 때)
        max_entropy = math.log2(45)
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _measure_pattern_diversity(self) -> float:
        """패턴 다양성 측정"""
        if len(self.recent_predictions) < 5:
            return 1.0
        
        pattern_types = set()
        
        for pred in list(self.recent_predictions)[-20:]:  # 최근 20개 예측
            # 홀짝 패턴
            odd_count = sum(1 for num in pred if num % 2 == 1)
            pattern_types.add(f"odd_{odd_count}")
            
            # 구간 분포 패턴
            sections = [0, 0, 0, 0]  # 1-10, 11-20, 21-30, 31-45
            for num in pred:
                if num <= 10: sections[0] += 1
                elif num <= 20: sections[1] += 1
                elif num <= 30: sections[2] += 1
                else: sections[3] += 1
            pattern_types.add(f"section_{tuple(sections)}")
        
        # 이론적 최대 패턴 수 대비 실제 패턴 수
        max_possible_patterns = min(20, 7 * 15)  # 홀짝 7가지 × 구간분포 조합 (단순화)
        return len(pattern_types) / max_possible_patterns
    
    def _update_diversity_metrics(self):
        """다양성 지표 업데이트"""
        self.diversity_metrics = {
            'health_score': self.calculate_diversity_health(),
            'unique_predictions': len(set(tuple(pred['numbers']) for pred in self.prediction_history)),
            'overused_numbers': len(self.get_overused_numbers()),
            'underused_numbers': len(self.get_underused_numbers()),
            'total_predictions': self.prediction_count
        }
    
    def _validate_prediction(self, numbers: List[int]) -> bool:
        """예측 유효성 검증"""
        if not isinstance(numbers, list):
            return False
        
        if len(numbers) != 6:
            return False
        
        if len(set(numbers)) != 6:  # 중복 체크
            return False
        
        for num in numbers:
            if not isinstance(num, int) or not (1 <= num <= 45):
                return False
        
        return True
    
    def get_diversity_report(self) -> Dict[str, Any]:
        """다양성 현황 리포트 반환"""
        if self.prediction_count == 0:
            return {"status": "insufficient_data"}
        
        return {
            "total_predictions": self.prediction_count,
            "unique_predictions": len(set(tuple(pred['numbers']) for pred in self.prediction_history)),
            "diversity_health": self.calculate_diversity_health(),
            "overused_numbers": self.get_overused_numbers(),
            "underused_numbers": self.get_underused_numbers(),
            "recent_uniqueness": self._measure_recent_uniqueness(),
            "distribution_entropy": self._measure_distribution_entropy(),
            "pattern_diversity": self._measure_pattern_diversity(),
            "number_usage_stats": dict(self.number_usage_count)
        }

class AdaptiveDiversityController:
    """적응형 다양성 제어기"""
    
    def __init__(self):
        self.diversity_settings = Config.DIVERSITY_SETTINGS.copy()
        self.adjustment_history = []
        
    def adjust_diversity_settings(self, current_health: float) -> Dict[str, Any]:
        """
        다양성 건강도에 따른 설정 동적 조정
        
        Args:
            current_health (float): 현재 다양성 건강도 (0.0~1.0)
            
        Returns:
            Dict[str, Any]: 조정된 설정
        """
        adjusted_settings = self.diversity_settings.copy()
        
        if current_health < 0.3:  # 다양성 심각 부족
            # 매우 엄격한 다양성 규칙 적용
            adjusted_settings['max_number_repeat_rate'] *= 0.6
            adjusted_settings['min_new_numbers_per_prediction'] += 2
            adjusted_settings['recent_avoidance_window'] += 10
            adjusted_settings['overuse_threshold'] *= 0.7
            
        elif current_health < 0.6:  # 다양성 부족
            # 엄격한 다양성 규칙 적용
            adjusted_settings['max_number_repeat_rate'] *= 0.8
            adjusted_settings['min_new_numbers_per_prediction'] += 1
            adjusted_settings['recent_avoidance_window'] += 5
            adjusted_settings['overuse_threshold'] *= 0.8
            
        elif current_health > 0.8:  # 다양성 충분
            # 다양성 규칙 완화
            adjusted_settings['max_number_repeat_rate'] *= 1.1
            adjusted_settings['min_new_numbers_per_prediction'] = max(1, 
                adjusted_settings['min_new_numbers_per_prediction'] - 1)
            adjusted_settings['recent_avoidance_window'] = max(5,
                adjusted_settings['recent_avoidance_window'] - 3)
        
        # 조정 이력 기록
        self.adjustment_history.append({
            'timestamp': datetime.now().isoformat(),
            'health_score': current_health,
            'adjustments': adjusted_settings
        })
        
        return adjusted_settings

class DiversityGuaranteedPredictor:
    """다양성이 보장된 예측기"""
    
    def __init__(self, base_predictor, history: PredictionHistory):
        """
        초기화
        
        Args:
            base_predictor: 기본 예측기 (ensemble_predictor 등)
            history (PredictionHistory): 예측 히스토리 관리자
        """
        self.base_predictor = base_predictor
        self.history = history
        self.diversity_controller = AdaptiveDiversityController()
        self.logger = self._setup_logger()
        
        # 기본 다양성 설정
        self.diversity_config = Config.DIVERSITY_SETTINGS.copy()
    
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
    
    def predict_with_diversity(self, count: int = 6) -> List[int]:
        """
        다양성이 보장된 예측 생성
        
        Args:
            count (int): 예측할 번호 개수
            
        Returns:
            List[int]: 다양성이 보장된 예측 번호
        """
        max_attempts = 100  # 무한루프 방지
        
        # 현재 다양성 건강도 확인 및 설정 조정
        current_health = self.history.calculate_diversity_health()
        adjusted_config = self.diversity_controller.adjust_diversity_settings(current_health)
        
        self.logger.info(f"다양성 건강도: {current_health:.3f}")
        
        for attempt in range(max_attempts):
            try:
                # 기본 예측기에서 후보 생성
                if hasattr(self.base_predictor, 'predict_ensemble'):
                    # EnsemblePredictor인 경우
                    prediction_result = self.base_predictor.predict_ensemble(count)
                    candidate_numbers = prediction_result['final_prediction']
                else:
                    # 다른 예측기인 경우
                    candidate_numbers = self.base_predictor.predict_numbers(count)
                
                # 다양성 검증
                if self._validate_diversity(candidate_numbers, adjusted_config):
                    self.history.add_prediction(candidate_numbers, {
                        'method': 'diversity_guaranteed',
                        'attempt': attempt + 1,
                        'diversity_health': current_health,
                        'config_used': adjusted_config
                    })
                    
                    self.logger.info(f"다양성 보장 예측 성공 (시도: {attempt + 1})")
                    return candidate_numbers
                
                # 다양성 규칙 위반 시 강제 조정
                adjusted_numbers = self._force_diversity(candidate_numbers, adjusted_config)
                if adjusted_numbers and self._validate_diversity(adjusted_numbers, adjusted_config):
                    self.history.add_prediction(adjusted_numbers, {
                        'method': 'diversity_forced',
                        'original_prediction': candidate_numbers,
                        'attempt': attempt + 1,
                        'diversity_health': current_health
                    })
                    
                    self.logger.info(f"다양성 강제 조정 성공 (시도: {attempt + 1})")
                    return adjusted_numbers
                
            except Exception as e:
                self.logger.warning(f"예측 시도 {attempt + 1} 실패: {e}")
                continue
        
        # 최후의 수단: 완전 다양성 중심 예측
        self.logger.warning("최대 시도 횟수 초과, 비상 다양성 예측 사용")
        fallback_numbers = self._generate_emergency_prediction(count)
        self.history.add_prediction(fallback_numbers, {
            'method': 'emergency_diversity',
            'diversity_health': current_health
        })
        
        return fallback_numbers
    
    def _validate_diversity(self, numbers: List[int], config: Dict[str, Any]) -> bool:
        """
        다양성 규칙 검증
        
        Args:
            numbers (List[int]): 검증할 번호 리스트
            config (Dict[str, Any]): 다양성 설정
            
        Returns:
            bool: 다양성 규칙 통과 여부
        """
        # 1. 완전 동일 예측 체크
        if self.history.is_duplicate_prediction(numbers):
            self.logger.debug("완전 동일 예측 탐지")
            return False
        
        # 2. 과도하게 사용된 번호 포함 체크
        overused_numbers = set(self.history.get_overused_numbers(config['overuse_threshold']))
        overused_in_prediction = sum(1 for num in numbers if num in overused_numbers)
        max_allowed_overused = len(numbers) * config['max_number_repeat_rate']
        
        if overused_in_prediction > max_allowed_overused:
            self.logger.debug(f"과도 사용 번호 초과: {overused_in_prediction}/{max_allowed_overused}")
            return False
        
        # 3. 최근 예측과의 중복도 체크
        if len(self.history.recent_predictions) >= config['recent_avoidance_window']:
            recent_numbers = set()
            recent_window = list(self.history.recent_predictions)[-config['recent_avoidance_window']:]
            
            for recent_pred in recent_window:
                recent_numbers.update(recent_pred)
            
            overlap_count = sum(1 for num in numbers if num in recent_numbers)
            max_allowed_overlap = len(numbers) - config['min_new_numbers_per_prediction']
            
            if overlap_count > max_allowed_overlap:
                self.logger.debug(f"최근 중복도 초과: {overlap_count}/{max_allowed_overlap}")
                return False
        
        # 4. 패턴 다양성 체크 (홀짝 비율)
        odd_count = sum(1 for num in numbers if num % 2 == 1)
        if odd_count < 1 or odd_count > 5:  # 극단적 홀짝 비율 방지
            self.logger.debug(f"극단적 홀짝 비율: {odd_count}/6")
            return False
        
        # 5. 구간 다양성 체크
        low_count = sum(1 for num in numbers if num <= 15)
        mid_count = sum(1 for num in numbers if 16 <= num <= 30)
        high_count = sum(1 for num in numbers if num >= 31)
        
        if low_count == 0 or mid_count == 0 or high_count == 0:  # 한 구간이 완전히 비는 것 방지
            if len(numbers) >= 6:  # 6개 선택 시에만 적용
                self.logger.debug(f"구간 편중: low={low_count}, mid={mid_count}, high={high_count}")
                return False
        
        return True
    
    def _force_diversity(self, original_numbers: List[int], config: Dict[str, Any]) -> Optional[List[int]]:
        """
        강제 다양성 적용
        
        Args:
            original_numbers (List[int]): 원본 예측 번호
            config (Dict[str, Any]): 다양성 설정
            
        Returns:
            Optional[List[int]]: 조정된 번호 리스트 (조정 불가능하면 None)
        """
        adjusted_numbers = original_numbers.copy()
        overused_numbers = set(self.history.get_overused_numbers(config['overuse_threshold']))
        underused_numbers = set(self.history.get_underused_numbers(config['underuse_threshold']))
        
        # 과도 사용 번호 제거 및 과소 사용 번호로 교체
        replacements_needed = []
        for i, num in enumerate(adjusted_numbers):
            if num in overused_numbers:
                replacements_needed.append(i)
        
        # 과소 사용 번호로 교체
        available_replacements = [num for num in underused_numbers 
                                if num not in adjusted_numbers]
        
        # 충분한 대체 번호가 없으면 일반 번호도 고려
        if len(available_replacements) < len(replacements_needed):
            all_available = [num for num in range(1, 46) 
                           if num not in adjusted_numbers and num not in overused_numbers]
            available_replacements.extend(all_available)
        
        # 교체 수행
        for i, replacement_idx in enumerate(replacements_needed):
            if i < len(available_replacements):
                adjusted_numbers[replacement_idx] = available_replacements[i]
        
        # 교체 후에도 다양성 규칙 위반시 None 반환
        if not self._validate_diversity(adjusted_numbers, config):
            return None
        
        return adjusted_numbers
    
    def _generate_emergency_prediction(self, count: int) -> List[int]:
        """
        최후 수단 예측 (완전 다양성 중심)
        
        Args:
            count (int): 예측할 번호 개수
            
        Returns:
            List[int]: 비상 예측 번호
        """
        # 1순위: 과소 사용 번호들
        underused_numbers = self.history.get_underused_numbers()
        
        if len(underused_numbers) >= count:
            # 과소 사용 번호만으로 구성
            selected = random.sample(underused_numbers, count)
        else:
            # 과소 사용 번호 + 보완
            selected = underused_numbers.copy()
            remaining_count = count - len(selected)
            
            # 과도 사용되지 않은 번호들로 보완
            overused_numbers = set(self.history.get_overused_numbers())
            available = [num for num in range(1, 46) 
                        if num not in selected and num not in overused_numbers]
            
            if len(available) >= remaining_count:
                selected.extend(random.sample(available, remaining_count))
            else:
                # 정말 최후의 수단: 아무 번호나
                all_available = [num for num in range(1, 46) if num not in selected]
                selected.extend(random.sample(all_available, remaining_count))
        
        # 구간 다양성 보정
        selected = self._ensure_section_diversity(selected, count)
        
        return sorted(selected)
    
    def _ensure_section_diversity(self, numbers: List[int], target_count: int) -> List[int]:
        """구간 다양성 보장"""
        if len(numbers) < 6:
            return numbers  # 6개 미만이면 그대로 반환
        
        # 현재 구간별 분포 확인
        low_numbers = [n for n in numbers if n <= 15]
        mid_numbers = [n for n in numbers if 16 <= n <= 30]
        high_numbers = [n for n in numbers if n >= 31]
        
        # 각 구간에 최소 1개씩 보장
        result = []
        
        # 각 구간에서 최소 1개씩 선택
        if low_numbers:
            result.append(random.choice(low_numbers))
        if mid_numbers:
            result.append(random.choice(mid_numbers))
        if high_numbers:
            result.append(random.choice(high_numbers))
        
        # 나머지 자리를 원본 번호들로 채움
        remaining = [n for n in numbers if n not in result]
        remaining_count = target_count - len(result)
        
        if len(remaining) >= remaining_count:
            result.extend(random.sample(remaining, remaining_count))
        else:
            result.extend(remaining)
            # 부족한 경우 랜덤 보완
            while len(result) < target_count:
                available = [i for i in range(1, 46) if i not in result]
                if available:
                    result.append(random.choice(available))
                else:
                    break
        
        return result[:target_count]
    
    def get_diversity_status(self) -> Dict[str, Any]:
        """현재 다양성 상태 반환"""
        health_score = self.history.calculate_diversity_health()
        
        return {
            'diversity_health': health_score,
            'health_level': self._classify_health_level(health_score),
            'total_predictions': self.history.prediction_count,
            'unique_predictions': len(set(tuple(pred['numbers']) for pred in self.history.prediction_history)),
            'overused_numbers': self.history.get_overused_numbers(),
            'underused_numbers': self.history.get_underused_numbers(),
            'recent_diversity': self.history._measure_recent_uniqueness(),
            'recommendations': self._generate_recommendations(health_score)
        }
    
    def _classify_health_level(self, health_score: float) -> str:
        """건강도 점수를 레벨로 분류"""
        if health_score >= 0.8:
            return 'excellent'
        elif health_score >= 0.6:
            return 'good'
        elif health_score >= 0.4:
            return 'fair'
        elif health_score >= 0.2:
            return 'poor'
        else:
            return 'critical'
    
    def _generate_recommendations(self, health_score: float) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if health_score < 0.3:
            recommendations.append("다양성이 매우 부족합니다. 강제 다양성 모드를 활성화하세요.")
        elif health_score < 0.6:
            recommendations.append("다양성 개선이 필요합니다. 최근 사용된 번호들을 의도적으로 회피하세요.")
        
        overused = self.history.get_overused_numbers()
        if len(overused) > 10:
            recommendations.append(f"과도 사용 번호가 많습니다: {overused[:5]}... 이들 번호 사용을 줄이세요.")
        
        underused = self.history.get_underused_numbers()
        if len(underused) > 15:
            recommendations.append(f"과소 사용 번호가 많습니다. 이들 번호를 더 활용하세요: {underused[:5]}...")
        
        if self.history._measure_recent_uniqueness() < 0.7:
            recommendations.append("최근 예측의 고유성이 부족합니다. 더 다양한 조합을 시도하세요.")
        
        return recommendations


# === 모듈 테스트 ===
if __name__ == "__main__":
    # 테스트용 더미 예측기
    class DummyPredictor:
        def predict_numbers(self, count=6):
            return random.sample(range(1, 46), count)
    
    print("=== 다양성 보장 시스템 테스트 ===")
    
    try:
        # 히스토리 및 예측기 초기화
        history = PredictionHistory(max_history=100)
        dummy_predictor = DummyPredictor()
        diversity_predictor = DiversityGuaranteedPredictor(dummy_predictor, history)
        
        # 다중 예측 테스트 (다양성 확인)
        predictions = []
        print("10회 연속 예측 수행:")
        
        for i in range(10):
            prediction = diversity_predictor.predict_with_diversity(6)
            predictions.append(prediction)
            print(f"예측 {i+1}: {prediction}")
        
        # 다양성 분석
        unique_predictions = len(set(tuple(sorted(p)) for p in predictions))
        print(f"\n다양성 분석:")
        print(f"총 예측 수: {len(predictions)}")
        print(f"고유 예측 수: {unique_predictions}")
        print(f"다양성 비율: {unique_predictions/len(predictions):.2f}")
        
        # 다양성 상태 확인
        status = diversity_predictor.get_diversity_status()
        print(f"\n다양성 건강도: {status['diversity_health']:.3f} ({status['health_level']})")
        print(f"과도 사용 번호: {status['overused_numbers'][:5]}")
        print(f"과소 사용 번호: {status['underused_numbers'][:5]}")
        
        if status['recommendations']:
            print("권장사항:")
            for rec in status['recommendations']:
                print(f"  - {rec}")
        
        print("✅ 다양성 보장 시스템 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
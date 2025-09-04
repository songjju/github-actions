"""
파일명: src/prediction/formula_engine.py
목적: 천재적 공식들을 통합 관리하는 공식 엔진
작성자: AI Assistant
작성일: 2025-09-04
버전: 1.0

주요 클래스/함수:
- FormulaEngine: 공식 엔진 메인 클래스
- FormulaRegistry: 공식 레지스트리 관리
- FormulaResult: 공식 실행 결과 클래스
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import sys
from datetime import datetime
import json
from collections import defaultdict, Counter
import math
import random

# 상위 디렉토리의 imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import Config, GeniusFormulaConstants

# 천재 공식 import (현재 구현된 것만)
try:
    from src.genius_formulas.formula_01_genius_insight import GeniusInsightFormula
except ImportError:
    GeniusInsightFormula = None

class FormulaResult:
    """공식 실행 결과를 담는 클래스"""
    
    def __init__(self, formula_name: str):
        self.formula_name = formula_name
        self.predictions = []
        self.confidence = 0.0
        self.method_info = {}
        self.execution_time = 0.0
        self.error = None
        self.success = True
        self.details = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'formula_name': self.formula_name,
            'predictions': self.predictions,
            'confidence': self.confidence,
            'method_info': self.method_info,
            'execution_time': self.execution_time,
            'error': self.error,
            'success': self.success,
            'details': self.details
        }

class BaseFormula(ABC):
    """기본 공식 추상 클래스"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(f"{__name__}.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, statistics: Dict[str, Any], count: int = 6) -> FormulaResult:
        """예측 수행 (추상 메서드)"""
        pass
    
    def validate_input(self, data: pd.DataFrame, statistics: Dict[str, Any]) -> bool:
        """입력 데이터 유효성 검사"""
        if data.empty:
            self.logger.warning(f"{self.name}: 데이터가 비어있습니다")
            return False
        
        if not isinstance(statistics, dict):
            self.logger.warning(f"{self.name}: 통계 데이터가 딕셔너리가 아닙니다")
            return False
        
        return True

class GeniusInsightFormulaAdapter(BaseFormula):
    """천재적 통찰 공식 어댑터"""
    
    def __init__(self):
        super().__init__("GeniusInsight", Config.FORMULA_WEIGHTS.get('genius_insight', 1.0))
        self.formula_instance = None
    
    def predict(self, data: pd.DataFrame, statistics: Dict[str, Any], count: int = 6) -> FormulaResult:
        """천재적 통찰 공식 예측"""
        result = FormulaResult(self.name)
        start_time = datetime.now()
        
        try:
            if not self.validate_input(data, statistics):
                result.success = False
                result.error = "입력 데이터 검증 실패"
                return result
            
            # GeniusInsightFormula가 사용 가능한 경우
            if GeniusInsightFormula:
                if not self.formula_instance:
                    self.formula_instance = GeniusInsightFormula(data, statistics)
                
                # 예측 수행
                predicted_numbers = self.formula_instance.predict_numbers(count)
                
                # 신뢰도 계산 (GI 점수 기반)
                gi_analysis = self.formula_instance.calculate_genius_insight()
                avg_gi_score = np.mean([gi_analysis['gi_scores'].get(num, 0) for num in predicted_numbers])
                confidence = min(max(avg_gi_score / 10.0, 0.1), 0.9)  # 0.1-0.9 범위로 정규화
                
                result.predictions = predicted_numbers
                result.confidence = confidence
                result.method_info = {
                    'formula': 'GI = (O × C × P × S) / (A + B)',
                    'components': gi_analysis.get('component_scores', {}),
                    'insights': gi_analysis.get('insights', [])
                }
                result.details = gi_analysis
                
            else:
                # 백업: 가중 랜덤 선택
                self.logger.warning("GeniusInsightFormula을 사용할 수 없어 백업 방식 사용")
                result.predictions = self._backup_prediction(data, statistics, count)
                result.confidence = 0.3
                result.method_info = {'method': 'backup_weighted_random'}
                result.details = {'note': 'GeniusInsightFormula import 실패로 백업 방식 사용'}
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.predictions = self._emergency_prediction(count)
            result.confidence = 0.1
            self.logger.error(f"천재적 통찰 공식 오류: {e}")
        
        finally:
            end_time = datetime.now()
            result.execution_time = (end_time - start_time).total_seconds()
        
        return result
    
    def _backup_prediction(self, data: pd.DataFrame, statistics: Dict[str, Any], count: int) -> List[int]:
        """백업 예측 방법 (빈도 기반 가중 선택)"""
        try:
            # 기본 통계에서 빈도 정보 추출
            frequency_data = statistics.get('frequency_analysis', {})
            frequency_count = frequency_data.get('frequency_count', {})
            
            if frequency_count:
                # 빈도를 확률로 변환
                total_freq = sum(frequency_count.values())
                probabilities = []
                
                for num in range(1, 46):
                    freq = frequency_count.get(num, 0)
                    # 빈도가 높을수록 높은 확률, 하지만 완전히 치우치지 않게 조정
                    prob = (freq + 1) / (total_freq + 45)  # 라플라스 스무딩
                    probabilities.append((num, prob))
                
                # 가중 선택
                return self._weighted_selection(probabilities, count)
            else:
                # 완전 백업: 균등 랜덤
                return random.sample(range(1, 46), count)
        
        except Exception:
            return self._emergency_prediction(count)
    
    def _weighted_selection(self, probabilities: List[Tuple[int, float]], count: int) -> List[int]:
        """가중치를 고려한 번호 선택"""
        numbers, weights = zip(*probabilities)
        selected = np.random.choice(numbers, size=count, replace=False, p=weights)
        return sorted(selected.tolist())
    
    def _emergency_prediction(self, count: int) -> List[int]:
        """응급 예측 (완전 랜덤)"""
        return sorted(random.sample(range(1, 46), count))

class StubFormula(BaseFormula):
    """미구현 공식들을 위한 스텁"""
    
    def __init__(self, name: str, weight: float = 0.0):
        super().__init__(name, weight)
    
    def predict(self, data: pd.DataFrame, statistics: Dict[str, Any], count: int = 6) -> FormulaResult:
        """스텁 예측 (가중 랜덤)"""
        result = FormulaResult(self.name)
        start_time = datetime.now()
        
        try:
            # 기본 통계 기반 가중 랜덤
            if self.weight > 0:
                result.predictions = self._statistical_random(data, statistics, count)
                result.confidence = min(self.weight * 0.3, 0.5)
            else:
                result.predictions = sorted(random.sample(range(1, 46), count))
                result.confidence = 0.1
            
            result.method_info = {
                'type': 'stub',
                'status': 'not_implemented',
                'weight': self.weight
            }
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.predictions = sorted(random.sample(range(1, 46), count))
            result.confidence = 0.05
        
        finally:
            end_time = datetime.now()
            result.execution_time = (end_time - start_time).total_seconds()
        
        return result
    
    def _statistical_random(self, data: pd.DataFrame, statistics: Dict[str, Any], count: int) -> List[int]:
        """통계 기반 랜덤 선택"""
        try:
            # 기본 제약 조건 적용
            selected = []
            attempts = 0
            max_attempts = 100
            
            while len(selected) < count and attempts < max_attempts:
                candidate = random.randint(1, 45)
                
                if candidate not in selected:
                    # 기본 제약 검사
                    if self._basic_constraints_check(selected + [candidate]):
                        selected.append(candidate)
                
                attempts += 1
            
            # 부족한 번호는 랜덤으로 채우기
            while len(selected) < count:
                candidate = random.randint(1, 45)
                if candidate not in selected:
                    selected.append(candidate)
            
            return sorted(selected)
        
        except Exception:
            return sorted(random.sample(range(1, 46), count))
    
    def _basic_constraints_check(self, numbers: List[int]) -> bool:
        """기본 제약 조건 확인"""
        if len(numbers) <= 1:
            return True
        
        # 너무 많은 연속 번호 방지
        sorted_nums = sorted(numbers)
        consecutive_count = 1
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] == sorted_nums[i-1] + 1:
                consecutive_count += 1
                if consecutive_count > 3:  # 3개 초과 연속 방지
                    return False
            else:
                consecutive_count = 1
        
        return True

class FormulaRegistry:
    """공식 레지스트리"""
    
    def __init__(self):
        self.formulas: Dict[str, BaseFormula] = {}
        self.logger = self._setup_logger()
        self._register_default_formulas()
    
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
    
    def _register_default_formulas(self):
        """기본 공식들 등록"""
        # 구현된 공식
        self.register_formula(GeniusInsightFormulaAdapter())
        
        # 미구현 공식들 (스텁으로 등록)
        stub_formulas = [
            ('MultiDimensional', Config.FORMULA_WEIGHTS.get('multi_dimensional', 0.8)),
            ('CreativeConnection', Config.FORMULA_WEIGHTS.get('creative_connection', 0.7)),
            ('ProblemRedefinition', Config.FORMULA_WEIGHTS.get('problem_redefinition', 0.6)),
            ('InnovativeSolution', Config.FORMULA_WEIGHTS.get('innovative_solution', 0.5)),
            ('InsightAmplification', Config.FORMULA_WEIGHTS.get('insight_amplification', 0.4)),
            ('ThinkingEvolution', Config.FORMULA_WEIGHTS.get('thinking_evolution', 0.3)),
            ('ComplexitySolution', Config.FORMULA_WEIGHTS.get('complexity_solution', 0.2)),
            ('IntuitiveLeap', Config.FORMULA_WEIGHTS.get('intuitive_leap', 0.4)),
            ('IntegratedWisdom', Config.FORMULA_WEIGHTS.get('integrated_wisdom', 1.0)),
        ]
        
        for name, weight in stub_formulas:
            self.register_formula(StubFormula(name, weight))
    
    def register_formula(self, formula: BaseFormula):
        """공식 등록"""
        self.formulas[formula.name] = formula
        self.logger.info(f"공식 등록: {formula.name} (가중치: {formula.weight})")
    
    def get_formula(self, name: str) -> Optional[BaseFormula]:
        """공식 조회"""
        return self.formulas.get(name)
    
    def get_all_formulas(self) -> Dict[str, BaseFormula]:
        """모든 공식 조회"""
        return self.formulas.copy()
    
    def get_active_formulas(self) -> Dict[str, BaseFormula]:
        """활성 공식들만 조회 (가중치 > 0)"""
        return {name: formula for name, formula in self.formulas.items() if formula.weight > 0}

class FormulaEngine:
    """천재적 공식 엔진 메인 클래스"""
    
    def __init__(self):
        self.registry = FormulaRegistry()
        self.logger = self._setup_logger()
        self.execution_history = []
        self.performance_stats = defaultdict(list)
    
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
    
    def apply_formulas(self, data: pd.DataFrame, statistics: Dict[str, Any], 
                      formula_names: Optional[List[str]] = None, 
                      prediction_count: int = 6) -> Dict[str, Any]:
        """
        공식들 적용하여 예측 수행
        
        Args:
            data: 로또 데이터
            statistics: 기초 통계 분석 결과
            formula_names: 사용할 공식 이름들 (None이면 모든 활성 공식)
            prediction_count: 예측할 번호 개수
            
        Returns:
            Dict: 예측 결과
        """
        self.logger.info("공식 엔진 실행 시작")
        start_time = datetime.now()
        
        try:
            # 사용할 공식들 결정
            if formula_names:
                formulas = {name: self.registry.get_formula(name) 
                           for name in formula_names 
                           if self.registry.get_formula(name)}
                formulas = {name: formula for name, formula in formulas.items() if formula}
            else:
                formulas = self.registry.get_active_formulas()
            
            if not formulas:
                raise ValueError("사용 가능한 공식이 없습니다")
            
            self.logger.info(f"사용할 공식: {list(formulas.keys())}")
            
            # 각 공식 실행
            formula_results = {}
            for name, formula in formulas.items():
                try:
                    self.logger.debug(f"{name} 공식 실행 중...")
                    result = formula.predict(data, statistics, prediction_count)
                    formula_results[name] = result
                    
                    # 성능 통계 기록
                    self.performance_stats[name].append({
                        'execution_time': result.execution_time,
                        'confidence': result.confidence,
                        'success': result.success,
                        'timestamp': datetime.now()
                    })
                    
                    self.logger.debug(f"{name}: 신뢰도={result.confidence:.3f}, "
                                    f"시간={result.execution_time:.3f}s")
                
                except Exception as e:
                    self.logger.error(f"{name} 공식 실행 오류: {e}")
                    error_result = FormulaResult(name)
                    error_result.success = False
                    error_result.error = str(e)
                    error_result.predictions = sorted(random.sample(range(1, 46), prediction_count))
                    error_result.confidence = 0.05
                    formula_results[name] = error_result
            
            # 결과 통합
            integrated_result = self._integrate_results(formula_results, prediction_count)
            
            # 실행 히스토리 기록
            execution_record = {
                'timestamp': start_time,
                'formulas_used': list(formulas.keys()),
                'execution_time': (datetime.now() - start_time).total_seconds(),
                'results': {name: result.to_dict() for name, result in formula_results.items()},
                'integrated_result': integrated_result
            }
            self.execution_history.append(execution_record)
            
            # 히스토리 관리 (최대 100개 유지)
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
            
            self.logger.info(f"공식 엔진 실행 완료: {len(formula_results)}개 공식, "
                           f"{(datetime.now() - start_time).total_seconds():.3f}초")
            
            return integrated_result
        
        except Exception as e:
            self.logger.error(f"공식 엔진 실행 오류: {e}")
            # 응급 결과 반환
            return {
                'predictions': [sorted(random.sample(range(1, 46), prediction_count))],
                'confidence': 0.1,
                'method': 'emergency_random',
                'error': str(e),
                'formula_results': {},
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    def _integrate_results(self, formula_results: Dict[str, FormulaResult], 
                          prediction_count: int) -> Dict[str, Any]:
        """공식 결과들을 통합"""
        try:
            successful_results = {name: result for name, result in formula_results.items() 
                                if result.success and result.predictions}
            
            if not successful_results:
                # 모든 공식이 실패한 경우
                return {
                    'predictions': [sorted(random.sample(range(1, 46), prediction_count))],
                    'confidence': 0.05,
                    'method': 'all_failed_fallback',
                    'formula_results': {name: result.to_dict() for name, result in formula_results.items()},
                    'integration_method': 'emergency'
                }
            
            # 가중치 기반 앙상블
            number_votes = defaultdict(float)
            total_weight = 0
            formula_contributions = {}
            
            for name, result in successful_results.items():
                formula = self.registry.get_formula(name)
                weight = formula.weight * result.confidence if formula else result.confidence
                total_weight += weight
                
                # 각 번호에 가중치 투표
                for number in result.predictions:
                    number_votes[number] += weight
                
                formula_contributions[name] = {
                    'weight': weight,
                    'confidence': result.confidence,
                    'predictions': result.predictions
                }
            
            # 상위 번호 선택
            sorted_votes = sorted(number_votes.items(), key=lambda x: x[1], reverse=True)
            final_prediction = [number for number, _ in sorted_votes[:prediction_count]]
            
            # 최종 신뢰도 계산 (가중 평균)
            if total_weight > 0:
                avg_confidence = sum(formula.weight * result.confidence 
                                   for formula_name, result in successful_results.items() 
                                   for formula in [self.registry.get_formula(formula_name)] 
                                   if formula) / total_weight
            else:
                avg_confidence = np.mean([result.confidence for result in successful_results.values()])
            
            return {
                'predictions': [final_prediction],
                'confidence': avg_confidence,
                'method': 'weighted_ensemble',
                'formula_results': {name: result.to_dict() for name, result in formula_results.items()},
                'integration_method': 'weighted_voting',
                'voting_details': {
                    'number_votes': dict(number_votes),
                    'formula_contributions': formula_contributions,
                    'total_weight': total_weight
                }
            }
        
        except Exception as e:
            self.logger.error(f"결과 통합 오류: {e}")
            return {
                'predictions': [sorted(random.sample(range(1, 46), prediction_count))],
                'confidence': 0.1,
                'method': 'integration_failed',
                'error': str(e),
                'formula_results': {name: result.to_dict() for name, result in formula_results.items()}
            }
    
    def get_available_formulas(self) -> List[str]:
        """사용 가능한 공식 목록"""
        return list(self.registry.formulas.keys())
    
    def get_formula_info(self, name: str) -> Optional[Dict[str, Any]]:
        """공식 정보 조회"""
        formula = self.registry.get_formula(name)
        if formula:
            recent_stats = self.performance_stats.get(name, [])
            return {
                'name': formula.name,
                'weight': formula.weight,
                'recent_executions': len(recent_stats),
                'avg_confidence': np.mean([s['confidence'] for s in recent_stats]) if recent_stats else 0,
                'avg_execution_time': np.mean([s['execution_time'] for s in recent_stats]) if recent_stats else 0,
                'success_rate': np.mean([s['success'] for s in recent_stats]) if recent_stats else 0
            }
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        summary = {
            'total_executions': len(self.execution_history),
            'formula_stats': {},
            'overall_stats': {
                'avg_execution_time': 0,
                'success_rate': 0
            }
        }
        
        # 공식별 통계
        for name, stats in self.performance_stats.items():
            if stats:
                summary['formula_stats'][name] = {
                    'executions': len(stats),
                    'avg_confidence': np.mean([s['confidence'] for s in stats]),
                    'avg_execution_time': np.mean([s['execution_time'] for s in stats]),
                    'success_rate': np.mean([s['success'] for s in stats])
                }
        
        # 전체 통계
        if self.execution_history:
            summary['overall_stats']['avg_execution_time'] = np.mean([
                h['execution_time'] for h in self.execution_history
            ])
        
        return summary

# === 사용 예제 ===
if __name__ == "__main__":
    # 테스트용 데이터 생성
    test_data = pd.DataFrame({
        'number_1': [1, 2, 3, 4, 5],
        'number_2': [7, 8, 9, 10, 11],
        'number_3': [14, 15, 16, 17, 18],
        'number_4': [21, 22, 23, 24, 25],
        'number_5': [28, 29, 30, 31, 32],
        'number_6': [35, 36, 37, 38, 39],
        'winning_numbers': [[1,7,14,21,28,35], [2,8,15,22,29,36], [3,9,16,23,30,37], [4,10,17,24,31,38], [5,11,18,25,32,39]]
    })
    
    test_stats = {
        'frequency_analysis': {
            'frequency_count': {i: random.randint(10, 30) for i in range(1, 46)}
        }
    }
    
    print("=== 공식 엔진 테스트 ===")
    
    # 공식 엔진 초기화
    engine = FormulaEngine()
    
    print(f"등록된 공식: {engine.get_available_formulas()}")
    
    # 공식 적용
    result = engine.apply_formulas(test_data, test_stats, prediction_count=6)
    
    print(f"\n예측 결과: {result['predictions'][0]}")
    print(f"신뢰도: {result['confidence']:.3f}")
    print(f"사용된 방법: {result['method']}")
    
    # 성능 요약
    performance = engine.get_performance_summary()
    print(f"\n성능 요약:")
    print(f"총 실행 횟수: {performance['total_executions']}")
    for name, stats in performance['formula_stats'].items():
        print(f"{name}: 신뢰도={stats['avg_confidence']:.3f}, 성공률={stats['success_rate']:.1%}")
    
    print("\n✅ 공식 엔진 테스트 완료")
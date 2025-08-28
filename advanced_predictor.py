import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
import hashlib
warnings.filterwarnings('ignore')

class AdvancedLottoPredictor:
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.scalers = {}
        
        # 중복 방지를 위한 추가
        self.prediction_history = []
        self.diversity_modes = ['conservative', 'moderate', 'aggressive']
        self.current_mode = 'moderate'
        
        # 자릿수별 다양성 관리 추가
        self.digit_patterns = {i: [] for i in range(6)}  # 각 자리별 패턴 추적
        self.position_weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]  # 자릿수별 가중치
    
    def _analyze_digit_patterns(self, predictions):
        """자릿수별 패턴 분석"""
        if not predictions:
            return {}
        
        patterns = {}
        for pos in range(6):
            digits_at_pos = []
            for pred in predictions:
                pred_str = str(pred).zfill(6)
                if len(pred_str) >= pos + 1:
                    digits_at_pos.append(int(pred_str[pos]))
            
            if digits_at_pos:
                patterns[f'pos_{pos+1}'] = {
                    'most_common': max(set(digits_at_pos), key=digits_at_pos.count),
                    'frequency': digits_at_pos.count(max(set(digits_at_pos), key=digits_at_pos.count)),
                    'unique_count': len(set(digits_at_pos)),
                    'recent_sequence': digits_at_pos[-5:]
                }
        
        return patterns

    def _check_digit_diversity(self, prediction):
        """자릿수별 다양성 검사"""
        pred_str = str(prediction).zfill(6)
        diversity_issues = []
        
        for pos in range(6):
            current_digit = int(pred_str[pos])
            recent_digits = self.digit_patterns[pos][-8:]  # 최근 8회 확인
            
            if len(recent_digits) >= 4:
                # 같은 숫자가 연속 4회 이상
                if recent_digits[-4:].count(current_digit) >= 4:
                    diversity_issues.append(f"자리{pos+1}: {current_digit} 연속반복")
                
                # 최근 8회 중 60% 이상 같은 숫자
                if recent_digits.count(current_digit) / len(recent_digits) > 0.6:
                    diversity_issues.append(f"자리{pos+1}: {current_digit} 과도편중")
        
        return len(diversity_issues) == 0, diversity_issues
    
    def _enhance_digit_diversity(self, base_prediction):
        """자릿수 다양성 향상"""
        pred_str = str(base_prediction).zfill(6)
        new_digits = list(pred_str)
        
        for pos in range(6):
            current_digit = int(new_digits[pos])
            recent_digits = self.digit_patterns[pos][-10:]
            
            # 현재 숫자의 최근 사용 빈도 확인
            if len(recent_digits) >= 3:
                frequency = recent_digits.count(current_digit)
                if frequency / len(recent_digits) > 0.5:  # 50% 이상이면 교체
                    # 가장 적게 사용된 숫자들 찾기
                    all_digits = list(range(10))
                    if pos == 0:  # 첫 번째 자리는 1-5만
                        all_digits = list(range(1, 6))
                    
                    digit_counts = {d: recent_digits.count(d) for d in all_digits}
                    min_count = min(digit_counts.values())
                    candidates = [d for d, count in digit_counts.items() if count == min_count]
                    
                    if candidates and current_digit not in candidates:
                        new_digit = np.random.choice(candidates)
                        new_digits[pos] = str(new_digit)
                        print(f"자릿수 다양성 개선: 위치{pos+1} {current_digit}→{new_digit}")
        
        return int(''.join(new_digits))
    
    def _update_digit_patterns(self, prediction):
        """자릿수별 패턴 기록 업데이트"""
        pred_str = str(prediction).zfill(6)
        
        for pos in range(6):
            digit = int(pred_str[pos])
            self.digit_patterns[pos].append(digit)
            
            # 최근 15개만 유지
            if len(self.digit_patterns[pos]) > 15:
                self.digit_patterns[pos] = self.digit_patterns[pos][-15:]

    def _generate_ensemble_seed(self):
        """앙상블 모델용 동적 시드 생성"""
        timestamp = int(time.time() * 1000) % 50000
        data_signature = hash(str(self.data.tail(3)['number'].sum())) % 10000
        history_count = len(self.prediction_history) % 1000
        
        return (timestamp + data_signature + history_count) % 100000

    def _add_model_noise(self, predictions, noise_level=0.02):
        """모델 예측에 제어된 노이즈 추가"""
        noise_seed = self._generate_ensemble_seed()
        np.random.seed(noise_seed)
        
        enhanced_predictions = {}
        for model_name, pred_value in predictions.items():
            if isinstance(pred_value, (int, float)):
                # 예측값의 일정 비율로 노이즈 추가
                noise = np.random.normal(0, abs(pred_value) * noise_level)
                enhanced_predictions[model_name] = pred_value + noise
            else:
                enhanced_predictions[model_name] = pred_value
                
        return enhanced_predictions

    def _check_prediction_uniqueness(self, prediction):
        """예측 고유성 검사"""
        if len(self.prediction_history) == 0:
            return True
            
        recent_predictions = self.prediction_history[-15:]  # 최근 15개 확인
        current_pred = str(prediction).zfill(6)
        
        for past_pred in recent_predictions:
            if abs(int(current_pred) - int(past_pred)) < 1000:  # 1000 이내 차이면 유사
                return False
                
        return True

    def _generate_diverse_alternative(self, base_prediction, attempt=0):
        """다양한 대안 예측 생성"""
        if attempt >= 8:
            # 최후의 수단: 완전 랜덤
            return np.random.randint(100000, 999999)
        
        base_num = int(str(base_prediction).zfill(6))
        
        # 다양성 전략들
        strategies = [
            lambda x: x + np.random.randint(5000, 15000),    # 큰 증가
            lambda x: x - np.random.randint(5000, 15000),    # 큰 감소  
            lambda x: x + np.random.randint(-3000, 3000),    # 중간 변화
            lambda x: self._digit_shuffle_strategy(x),       # 자릿수 섞기
            lambda x: self._pattern_shift_strategy(x),       # 패턴 이동
            lambda x: self._frequency_based_strategy(x),     # 빈도 기반
        ]
        
        strategy = strategies[attempt % len(strategies)]
        new_prediction = strategy(base_num)
        
        # 범위 조정
        new_prediction = max(100000, min(999999, new_prediction))
        
        # 재귀적 고유성 검사
        if not self._check_prediction_uniqueness(new_prediction):
            return self._generate_diverse_alternative(base_prediction, attempt + 1)
            
        return new_prediction

    def _digit_shuffle_strategy(self, number):
        """자릿수 섞기 전략"""
        digits = list(str(number).zfill(6))
        
        # 2-3개 자릿수 위치 변경
        positions = np.random.choice(6, size=3, replace=False)
        shuffled_digits = [digits[i] for i in positions]
        np.random.shuffle(shuffled_digits)
        
        for i, pos in enumerate(positions):
            digits[pos] = shuffled_digits[i]
            
        return int(''.join(digits))

    def _pattern_shift_strategy(self, number):
        """패턴 이동 전략"""
        digits = [int(d) for d in str(number).zfill(6)]
        
        # 특정 패턴 적용
        patterns = [
            lambda d: [(x + 1) % 10 for x in d],  # 모든 자릿수 +1
            lambda d: [(x + 2) % 10 for x in d],  # 모든 자릿수 +2
            lambda d: [x if i % 2 == 0 else (x + 3) % 10 for i, x in enumerate(d)],  # 짝수 위치만 +3
        ]
        
        pattern = np.random.choice(patterns)
        new_digits = pattern(digits)
        
        return int(''.join(map(str, new_digits)))

    def _frequency_based_strategy(self, number):
        """빈도 기반 전략"""
        if len(self.data) > 20:
            # 최근 데이터에서 가장 적게 나온 범위 찾기
            recent_numbers = self.data.tail(20)['number'].astype(int)
            
            # 10만 단위별 빈도 계산
            ranges = {}
            for num in recent_numbers:
                range_key = num // 100000
                ranges[range_key] = ranges.get(range_key, 0) + 1
            
            # 가장 적게 나온 범위 선택
            min_count = min(ranges.values()) if ranges else 1
            least_frequent_ranges = [r for r, count in ranges.items() if count == min_count]
            
            if least_frequent_ranges:
                chosen_range = np.random.choice(least_frequent_ranges)
                base = chosen_range * 100000
                return base + np.random.randint(0, 99999)
        
        return number

    def predict_ensemble(self):
        """개선된 앙상블 예측 - 중복 방지 포함"""
        if not self.models:
            print("No models trained, returning diverse random prediction")
            # 다양한 랜덤 예측 생성
            base_random = np.random.randint(100000, 999999)
            if not self._check_prediction_uniqueness(base_random):
                base_random = self._generate_diverse_alternative(base_random)
            
            result = {
                'ensemble_prediction': str(base_random).zfill(6),
                'model_predictions': {},
                'confidence': 50.0,
                'diversity_applied': True
            }
            
            self.prediction_history.append(str(base_random).zfill(6))
            return result
        
        result = self.prepare_sequence_data()
        if result[0] is None:
            return self._generate_fallback_prediction()
            
        X, _, _ = result
        
        if len(X) == 0:
            return self._generate_fallback_prediction()
        
        X_last = X[-1:] if len(X) > 0 else None
        raw_predictions = {}
        
        # 각 모델로 예측 (노이즈 추가)
        if 'lstm' in self.models and X_last is not None:
            try:
                lstm_pred = self.models['lstm'].predict(X_last, verbose=0)[0]
                raw_predictions['lstm'] = self.scalers['number'].inverse_transform([[lstm_pred]])[0, 0]
            except:
                pass
        
        if 'xgboost' in self.models and X_last is not None:
            try:
                X_flat = X_last.reshape(1, -1)
                xgb_pred = self.models['xgboost'].predict(X_flat)[0]
                raw_predictions['xgboost'] = self.scalers['number'].inverse_transform([[xgb_pred]])[0, 0]
            except:
                pass
        
        if not raw_predictions:
            return self._generate_fallback_prediction()
        
        # 모델 예측에 다양성 노이즈 추가
        enhanced_predictions = self._add_model_noise(raw_predictions, noise_level=0.03)
        
        # 가중 평균 계산
        weights = {'lstm': 0.6, 'xgboost': 0.4}
        weighted_sum = sum(enhanced_predictions.get(model, 0) * weights.get(model, 0.5) 
                          for model in enhanced_predictions.keys())
        total_weight = sum(weights.get(model, 0.5) for model in enhanced_predictions.keys())
        
        base_prediction = int(weighted_sum / total_weight) if total_weight > 0 else 500000
        base_prediction = base_prediction % 1000000
        
        # 고유성 검사 및 대안 생성
        if not self._check_prediction_uniqueness(base_prediction):
            print(f"중복 감지됨, 대안 생성 중: {base_prediction}")
            final_prediction = self._generate_diverse_alternative(base_prediction)
            diversity_applied = True
            confidence_adjustment = -5  # 신뢰도 약간 감소
        else:
            final_prediction = base_prediction
            diversity_applied = False
            confidence_adjustment = 0
        
        # 예측 기록에 추가
        self.prediction_history.append(str(final_prediction).zfill(6))
        
        # 최근 20개만 유지
        if len(self.prediction_history) > 20:
            self.prediction_history = self.prediction_history[-20:]
        
        base_confidence = self._calculate_ensemble_confidence(enhanced_predictions)
        final_confidence = max(50, base_confidence + confidence_adjustment)
        
        result = {
            'ensemble_prediction': str(final_prediction).zfill(6),
            'model_predictions': {k: str(int(v) % 1000000).zfill(6) 
                                for k, v in enhanced_predictions.items()},
            'confidence': round(final_confidence, 2),
            'diversity_applied': diversity_applied,
            'diversity_mode': self.current_mode
        }
        
        print(f"앙상블 예측 완료: {result['ensemble_prediction']} (다양성: {diversity_applied})")
        return result
    
    def get_digit_diversity_stats(self):
        """자릿수별 다양성 통계"""
        stats = {}
        
        for pos in range(6):
            recent_digits = self.digit_patterns[pos][-10:]
            if not recent_digits:
                stats[f'position_{pos+1}'] = {'diversity_score': 100, 'status': 'no_data'}
                continue
            
            unique_count = len(set(recent_digits))
            max_possible = 5 if pos == 0 else 10  # 첫 자리는 1-5만 가능
            diversity_score = (unique_count / min(max_possible, len(recent_digits))) * 100
            
            # 가장 빈번한 숫자 찾기
            most_common = max(set(recent_digits), key=recent_digits.count)
            frequency = recent_digits.count(most_common)
            
            stats[f'position_{pos+1}'] = {
                'diversity_score': round(diversity_score, 1),
                'unique_digits': unique_count,
                'most_common_digit': most_common,
                'most_common_frequency': frequency,
                'recent_sequence': recent_digits[-5:],
                'status': 'excellent' if diversity_score >= 80 else 'good' if diversity_score >= 60 else 'needs_improvement'
            }
        
        # 전체 평균
        all_scores = [stats[f'position_{i+1}']['diversity_score'] for i in range(6)]
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 100
        
        stats['overall'] = {
            'diversity_score': round(overall_score, 1),
            'status': 'excellent' if overall_score >= 80 else 'good' if overall_score >= 60 else 'needs_improvement'
        }
        
        return stats

    def _generate_fallback_prediction(self):
        """폴백 예측 생성"""
        fallback_num = np.random.randint(100000, 999999)
        
        # 고유성 확인
        if not self._check_prediction_uniqueness(fallback_num):
            fallback_num = self._generate_diverse_alternative(fallback_num)
        
        self.prediction_history.append(str(fallback_num).zfill(6))
        
        return {
            'ensemble_prediction': str(fallback_num).zfill(6),
            'model_predictions': {'fallback': str(fallback_num).zfill(6)},
            'confidence': 45.0,
            'diversity_applied': True,
            'diversity_mode': 'fallback'
        }

    def set_diversity_mode(self, mode):
        """다양성 모드 설정"""
        if mode in self.diversity_modes:
            self.current_mode = mode
            print(f"다양성 모드 변경: {mode}")
            
            # 모드별 노이즈 레벨 조정
            if mode == 'conservative':
                self.noise_level = 0.01
            elif mode == 'moderate':
                self.noise_level = 0.03
            elif mode == 'aggressive':
                self.noise_level = 0.05
        else:
            print(f"유효하지 않은 모드: {mode}. 사용 가능: {self.diversity_modes}")

    def get_diversity_stats(self):
        """다양성 통계 반환"""
        if len(self.prediction_history) < 3:
            return {"message": "예측 기록이 부족합니다"}
        
        # 최근 예측들의 차이 계산
        recent = [int(p) for p in self.prediction_history[-10:]]
        differences = []
        
        for i in range(1, len(recent)):
            diff = abs(recent[i] - recent[i-1])
            differences.append(diff)
        
        avg_difference = np.mean(differences) if differences else 0
        diversity_score = min(100, (avg_difference / 10000) * 100)  # 0-100 스케일
        
        return {
            'diversity_score': round(diversity_score, 2),
            'avg_difference': round(avg_difference, 0),
            'recent_predictions': self.prediction_history[-5:],
            'current_mode': self.current_mode,
            'total_predictions': len(self.prediction_history)
        }

    def clear_prediction_history(self):
        """예측 기록 초기화"""
        self.prediction_history = []
        print("예측 기록이 초기화되었습니다")

    def create_lstm_model(self, input_shape):
        """양방향 LSTM 모델"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def prepare_sequence_data(self, sequence_length=10):
        """시계열 데이터 준비"""
        if len(self.data) < sequence_length + 1:
            print(f"Warning: Not enough data for sequence length {sequence_length}")
            sequence_length = min(5, len(self.data) - 1)
        
        numbers = self.data['number'].astype(int).values
        jos = self.data['jo'].values
        
        # 스케일링
        number_scaler = MinMaxScaler(feature_range=(0, 1))
        numbers_scaled = number_scaler.fit_transform(numbers.reshape(-1, 1)).flatten()
        
        X, y, y_jo = [], [], []
        
        for i in range(len(numbers) - sequence_length):
            # 시퀀스 특징 추출
            seq_features = []
            for j in range(i, i + sequence_length):
                num_str = str(numbers[j]).zfill(6)
                
                # 각 시점의 특징
                features = [
                    numbers_scaled[j],  # 정규화된 번호
                    jos[j] / 5.0,  # 정규화된 조
                    sum(int(d) for d in num_str) / 54.0,  # 자릿수 합 (최대 54)
                    sum(1 for d in num_str if int(d) % 2) / 6.0,  # 홀수 비율
                    len(set(num_str)) / 6.0,  # 유니크 자릿수 비율
                ]
                seq_features.append(features)
            
            X.append(seq_features)
            y.append(numbers_scaled[i + sequence_length])
            y_jo.append(jos[i + sequence_length])
        
        self.scalers['number'] = number_scaler
        
        if len(X) == 0:
            print("Warning: No sequences created")
            return None, None, None
            
        return np.array(X), np.array(y), np.array(y_jo)
    
    def train_advanced_models(self):
        """모든 고급 모델 학습"""
        result = self.prepare_sequence_data()
        
        if result[0] is None:
            print("Skipping advanced model training due to insufficient data")
            return self.models
            
        X, y, y_jo = result
        
        print(f"Data shape: X={X.shape}, y={y.shape}")
        
        if len(X) < 10:
            print("Warning: Very limited data for training advanced models")
            return self.models
        
        # 데이터 분할
        split_idx = int(len(X) * 0.8)
        split_idx = max(split_idx, 1)  # 최소 1개는 훈련 데이터로
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 1. LSTM 모델 (충분한 데이터가 있을 때만)
        if len(X_train) > 20 and X.shape[1] > 0 and X.shape[2] > 0:
            try:
                print("Training LSTM model...")
                lstm_model = self.create_lstm_model((X.shape[1], X.shape[2]))
                
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                # 검증 데이터가 충분한 경우에만 validation_data 사용
                if len(X_test) > 0:
                    lstm_model.fit(
                        X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=50,
                        batch_size=min(32, len(X_train)),
                        callbacks=[early_stop],
                        verbose=0
                    )
                else:
                    lstm_model.fit(
                        X_train, y_train,
                        epochs=50,
                        batch_size=min(32, len(X_train)),
                        verbose=0
                    )
                
                self.models['lstm'] = lstm_model
                print("LSTM model trained successfully")
            except Exception as e:
                print(f"LSTM training failed: {e}")
        else:
            print("Skipping LSTM due to insufficient data")
        
        # 2. XGBoost 모델
        try:
            print("Training XGBoost model...")
            X_flat = X.reshape(X.shape[0], -1)
            X_train_flat = X_flat[:split_idx]
            X_test_flat = X_flat[split_idx:] if split_idx < len(X_flat) else X_flat[:1]
            
            xgb_model = XGBRegressor(
                n_estimators=100,  # 데이터가 적으므로 줄임
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            xgb_model.fit(X_train_flat, y_train, 
                         eval_set=[(X_test_flat, y_test)] if len(y_test) > 0 else None,
                         early_stopping_rounds=10 if len(y_test) > 0 else None,
                         verbose=0)
            
            self.models['xgboost'] = xgb_model
            print("XGBoost model trained successfully")
        except Exception as e:
            print(f"XGBoost training failed: {e}")
        
        return self.models
    
    def _calculate_ensemble_confidence(self, predictions):
        """앙상블 신뢰도 계산"""
        if len(predictions) < 1:
            return 50.0
        
        if len(predictions) == 1:
            return 70.0
        
        values = list(predictions.values())
        std_dev = np.std(values)
        mean_val = np.mean(values)
        
        # 변동계수 (낮을수록 모델들이 일치)
        cv = std_dev / mean_val if mean_val != 0 else 1
        confidence = max(50, min(95, (1 - cv) * 100))
        
        return round(confidence, 2)
    
    def analyze_deep_patterns(self):
        """심층 패턴 분석"""
        patterns = {}
        
        try:
            # 수치 패턴
            numbers = self.data['number'].astype(str).str.zfill(6)
            
            # 피보나치 수열과의 관계
            fib = [0, 1, 1, 2, 3, 5, 8]
            fib_patterns = []
            for num in numbers:
                fib_count = sum(1 for d in num if int(d) in fib)
                fib_patterns.append(fib_count)
            patterns['fibonacci_avg'] = np.mean(fib_patterns) if fib_patterns else 0
            
            # 소수 패턴
            primes = [2, 3, 5, 7]
            prime_patterns = []
            for num in numbers:
                prime_count = sum(1 for d in num if int(d) in primes)
                prime_patterns.append(prime_count)
            patterns['prime_avg'] = np.mean(prime_patterns) if prime_patterns else 0
            
            # 주기성 (간단한 분석)
            if len(self.data) > 10:
                number_series = self.data['number'].astype(int).values
                # 간단한 이동평균으로 트렌드 제거
                window = min(10, len(number_series) // 2)
                if window > 1:
                    trend = pd.Series(number_series).rolling(window=window, center=True).mean()
                    detrended = number_series - trend.fillna(method='bfill').fillna(method='ffill').values
                    
                    # 자기상관으로 주기 추정
                    autocorr = np.correlate(detrended, detrended, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    # 첫 번째 피크 찾기
                    peaks = []
                    for i in range(1, min(len(autocorr)-1, 50)):
                        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                            peaks.append(i)
                    
                    patterns['dominant_period'] = peaks[0] if peaks else 0
                else:
                    patterns['dominant_period'] = 0
            else:
                patterns['dominant_period'] = 0
                
        except Exception as e:
            print(f"Pattern analysis error: {e}")
            patterns = {
                'fibonacci_avg': 3.0,
                'prime_avg': 2.5,
                'dominant_period': 7.0
            }
        
        return patterns
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import numpy as np
import pandas as pd
from matplotlib import font_manager, rc

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class LottoVisualizer:
    def __init__(self, data=None):
        self.data = data
        sns.set_theme(style="whitegrid")
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
    def create_jo_distribution_chart(self):
        """조 분포 차트"""
        try:
            if self.data is None or self.data.empty:
                return self._create_sample_jo_chart()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # 전체 분포
            if 'jo' in self.data.columns:
                jo_counts = self.data['jo'].value_counts().sort_index()
            else:
                # jo 컬럼이 없으면 샘플 데이터 생성
                jo_counts = pd.Series([50, 60, 45, 55, 40], index=[1, 2, 3, 4, 5])
            
            bars = ax1.bar(jo_counts.index, jo_counts.values, color=self.colors)
            
            for bar, count in zip(bars, jo_counts.values):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            ax1.set_xlabel('Jo Number', fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title('Overall Jo Distribution', fontsize=14, fontweight='bold')
            
            # 최근 100회차 분포
            if len(self.data) >= 100:
                recent_jo = self.data.tail(100)['jo'].value_counts().sort_index()
            else:
                recent_jo = jo_counts
                
            bars2 = ax2.bar(recent_jo.index, recent_jo.values, color=self.colors, alpha=0.7)
            
            for bar, count in zip(bars2, recent_jo.values):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{count}', ha='center', va='bottom', fontweight='bold')
            
            ax2.set_xlabel('Jo Number', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title('Recent 100 Draws Jo Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"조 분포 차트 생성 중 오류: {e}")
            return self._create_sample_jo_chart()
    
    def create_number_trend_chart(self):
        """번호 추세 차트 - pivot_table 에러 해결"""
        try:
            if self.data is None or self.data.empty:
                return self._create_sample_trend_chart()
                
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            
            # 최근 50회차
            recent = self.data.tail(50).copy()
            
            # 데이터 정리 및 검증
            if 'number' not in recent.columns:
                print("number 컬럼이 없습니다.")
                return self._create_sample_trend_chart()
            
            if 'draw_no' not in recent.columns:
                recent['draw_no'] = range(len(recent))
                
            # number를 정수로 변환 (오류 처리 포함)
            try:
                recent['number_int'] = pd.to_numeric(recent['number'], errors='coerce')
                # NaN 값 제거
                recent = recent.dropna(subset=['number_int'])
                
                if recent.empty:
                    return self._create_sample_trend_chart()
                    
            except Exception as e:
                print(f"number 컬럼 변환 오류: {e}")
                return self._create_sample_trend_chart()
            
            # 번호 추세
            ax1.plot(recent['draw_no'], recent['number_int'], 
                    'o-', color='#45B7D1', markersize=6, linewidth=2, alpha=0.8)
            
            # 이동평균선
            for window, color in [(5, '#FF6B6B'), (10, '#4ECDC4'), (20, '#FECA57')]:
                if len(recent) >= window:
                    ma = recent['number_int'].rolling(window=window, center=True).mean()
                    ax1.plot(recent['draw_no'], ma, color=color, linewidth=2.5, 
                            label=f'{window}-draw MA', alpha=0.8)
            
            ax1.set_ylabel('Winning Number', fontsize=12)
            ax1.set_title('Recent 50 Draws Trend', fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # 조별 추세 (pivot_table 에러 해결)
            try:
                if 'jo' not in recent.columns:
                    # jo 컬럼이 없으면 number의 첫 번째 자릿수로 생성
                    recent['jo'] = recent['number'].astype(str).str[0].astype(int)
                
                # pivot_table 대신 안전한 방법 사용
                jo_data = []
                for _, row in recent.iterrows():
                    jo_data.append({
                        'draw_no': row['draw_no'],
                        'jo': row['jo']
                    })
                
                # 각 조별로 나타난 회차 표시
                jo_colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4', 5: '#FECA57'}
                
                for jo in range(1, 6):
                    jo_draws = [item['draw_no'] for item in jo_data if item['jo'] == jo]
                    if jo_draws:
                        ax2.scatter(jo_draws, [jo] * len(jo_draws),
                                  s=100, alpha=0.8, color=jo_colors.get(jo, '#666666'),
                                  label=f'Jo {jo}')
                
                ax2.set_xlabel('Draw Number', fontsize=12)
                ax2.set_ylabel('Jo Number', fontsize=12)
                ax2.set_title('Jo Appearance Pattern', fontsize=14, fontweight='bold')
                ax2.set_yticks(range(1, 6))
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='upper right')
                
            except Exception as jo_error:
                print(f"조별 추세 차트 생성 오류: {jo_error}")
                # 조별 추세가 실패하면 간단한 메시지 표시
                ax2.text(0.5, 0.5, 'Jo trend data not available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_xlabel('Draw Number', fontsize=12)
                ax2.set_ylabel('Jo Number', fontsize=12)
                ax2.set_title('Jo Appearance Pattern', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"번호 추세 차트 생성 중 전체 오류: {e}")
            return self._create_sample_trend_chart()
    
    def create_advanced_heatmap(self):
        """고급 히트맵"""
        try:
            if self.data is None or self.data.empty:
                return self._create_sample_heatmap()
                
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. 자리수별 숫자 히트맵
            try:
                digit_counts = np.zeros((10, 6))
                for num in self.data['number'].astype(str).str.zfill(6):
                    for pos, digit in enumerate(num):
                        if pos < 6 and digit.isdigit():
                            digit_counts[int(digit)][pos] += 1
                
                sns.heatmap(digit_counts, annot=True, fmt='.0f', cmap='YlOrRd',
                           xticklabels=[f'Pos {i+1}' for i in range(6)],
                           yticklabels=range(10), ax=axes[0,0], cbar_kws={'label': 'Count'})
                axes[0,0].set_title('Digit Frequency by Position', fontsize=14, fontweight='bold')
            except Exception as e:
                axes[0,0].text(0.5, 0.5, f'Digit analysis error\n{str(e)[:30]}...', 
                              ha='center', va='center', transform=axes[0,0].transAxes)
            
            # 2. 조별 번호 범위 히트맵
            try:
                jo_ranges = np.zeros((5, 10))
                for jo in range(1, 6):
                    if 'jo' in self.data.columns:
                        jo_numbers = self.data[self.data['jo'] == jo]['number'].astype(int)
                    else:
                        # jo 컬럼이 없으면 number 첫 자리로 판단
                        first_digits = self.data['number'].astype(str).str[0].astype(int)
                        jo_numbers = self.data[first_digits == jo]['number'].astype(int)
                    
                    for num in jo_numbers:
                        range_idx = min(num // 100000, 9)  # 범위 벗어남 방지
                        jo_ranges[jo-1][range_idx] += 1
                
                sns.heatmap(jo_ranges, annot=True, fmt='.0f', cmap='Blues',
                           xticklabels=[f'{i*100000}-{(i+1)*100000-1}' for i in range(10)],
                           yticklabels=[f'Jo {i}' for i in range(1, 6)], ax=axes[0,1])
                axes[0,1].set_title('Number Range Distribution by Jo', fontsize=14, fontweight='bold')
                axes[0,1].tick_params(axis='x', rotation=45)
            except Exception as e:
                axes[0,1].text(0.5, 0.5, f'Range analysis error\n{str(e)[:30]}...', 
                              ha='center', va='center', transform=axes[0,1].transAxes)
            
            # 3. 월별 조 분포
            try:
                if 'draw_date' in self.data.columns and 'jo' in self.data.columns:
                    # draw_date를 datetime으로 변환
                    self.data['draw_date'] = pd.to_datetime(self.data['draw_date'], errors='coerce')
                    monthly_jo = pd.crosstab(self.data['draw_date'].dt.month, self.data['jo'])
                    
                    sns.heatmap(monthly_jo, annot=True, fmt='.0f', cmap='Greens',
                               xticklabels=[f'Jo {i}' for i in range(1, 6)],
                               yticklabels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_jo)],
                               ax=axes[1,0])
                    axes[1,0].set_title('Monthly Jo Distribution', fontsize=14, fontweight='bold')
                else:
                    axes[1,0].text(0.5, 0.5, 'Monthly data not available', 
                                  ha='center', va='center', transform=axes[1,0].transAxes)
            except Exception as e:
                axes[1,0].text(0.5, 0.5, f'Monthly analysis error\n{str(e)[:30]}...', 
                              ha='center', va='center', transform=axes[1,0].transAxes)
            
            # 4. 상관관계 히트맵
            try:
                corr_data = pd.DataFrame()
                
                if 'jo' in self.data.columns:
                    corr_data['jo'] = pd.to_numeric(self.data['jo'], errors='coerce')
                else:
                    corr_data['jo'] = pd.to_numeric(self.data['number'].astype(str).str[0], errors='coerce')
                
                corr_data['number'] = pd.to_numeric(self.data['number'], errors='coerce')
                corr_data['digit_sum'] = self.data['number'].apply(
                    lambda x: sum(int(d) for d in str(x).zfill(6) if d.isdigit())
                )
                corr_data['odd_count'] = self.data['number'].apply(
                    lambda x: sum(1 for d in str(x).zfill(6) if d.isdigit() and int(d) % 2)
                )
                corr_data['unique_digits'] = self.data['number'].apply(
                    lambda x: len(set(str(x).zfill(6)))
                )
                
                # NaN 값 제거
                corr_data = corr_data.dropna()
                
                if not corr_data.empty:
                    sns.heatmap(corr_data.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                               center=0, ax=axes[1,1], square=True)
                    axes[1,1].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
                else:
                    axes[1,1].text(0.5, 0.5, 'Correlation data not available', 
                                  ha='center', va='center', transform=axes[1,1].transAxes)
                    
            except Exception as e:
                axes[1,1].text(0.5, 0.5, f'Correlation analysis error\n{str(e)[:30]}...', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"히트맵 생성 중 오류: {e}")
            return self._create_sample_heatmap()
    
    def create_prediction_confidence_chart(self, predictions):
        """예측 신뢰도 차트"""
        try:
            if not predictions:
                predictions = {
                    'Model A': {'confidence': 75.5},
                    'Model B': {'confidence': 68.2},
                    'Model C': {'confidence': 72.8}
                }
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            models = list(predictions.keys())
            confidences = [predictions[model].get('confidence', 50) for model in models]
            
            bars = ax.bar(models, confidences, color=self.colors[:len(models)])
            
            for bar, conf in zip(bars, confidences):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{conf:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylim(0, 100)
            ax.set_ylabel('Confidence (%)', fontsize=12)
            ax.set_title('Model Prediction Confidence', fontsize=14, fontweight='bold')
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Good Confidence')
            ax.legend()
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            print(f"예측 신뢰도 차트 생성 중 오류: {e}")
            return self._create_sample_confidence_chart()
    
    def _create_sample_jo_chart(self):
        """샘플 조 분포 차트"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 샘플 데이터
        jo_counts = pd.Series([45, 52, 38, 48, 42], index=[1, 2, 3, 4, 5])
        
        for ax, title in [(ax1, 'Overall Jo Distribution (Sample)'), 
                         (ax2, 'Recent Jo Distribution (Sample)')]:
            bars = ax.bar(jo_counts.index, jo_counts.values, color=self.colors)
            for bar, count in zip(bars, jo_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{count}', ha='center', va='bottom', fontweight='bold')
            ax.set_xlabel('Jo Number', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_sample_trend_chart(self):
        """샘플 트렌드 차트"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 샘플 번호 데이터
        draws = range(671, 721)
        numbers = np.random.randint(100000, 600000, 50)
        
        ax1.plot(draws, numbers, 'o-', color='#45B7D1', markersize=6, linewidth=2, alpha=0.8)
        ax1.set_ylabel('Winning Number', fontsize=12)
        ax1.set_title('Sample Number Trend', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 샘플 조 패턴
        jo_colors = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1', 4: '#96CEB4', 5: '#FECA57'}
        for jo in range(1, 6):
            sample_draws = np.random.choice(draws, 8, replace=False)
            ax2.scatter(sample_draws, [jo] * len(sample_draws),
                       s=100, alpha=0.8, color=jo_colors[jo], label=f'Jo {jo}')
        
        ax2.set_xlabel('Draw Number', fontsize=12)
        ax2.set_ylabel('Jo Number', fontsize=12)
        ax2.set_title('Sample Jo Pattern', fontsize=14, fontweight='bold')
        ax2.set_yticks(range(1, 6))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_sample_heatmap(self):
        """샘플 히트맵"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 간단한 샘플 히트맵
        sample_data = np.random.rand(10, 10)
        sns.heatmap(sample_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Sample Heatmap (Data not available)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_sample_confidence_chart(self):
        """샘플 신뢰도 차트"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = ['Model A', 'Model B', 'Model C']
        confidences = [75.5, 68.2, 72.8]
        
        bars = ax.bar(models, confidences, color=self.colors[:3])
        
        for bar, conf in zip(bars, confidences):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{conf:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, 100)
        ax.set_ylabel('Confidence (%)', fontsize=12)
        ax.set_title('Sample Model Confidence', fontsize=14, fontweight='bold')
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Good Confidence')
        ax.legend()
        
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig):
        """Figure를 base64로 변환"""
        try:
            buffer = BytesIO()
            fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Figure를 base64로 변환 중 오류: {e}")
            plt.close(fig)
            # 빈 이미지 반환
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
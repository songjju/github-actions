#!/usr/bin/env python3
"""
파일명: main.py
목적: 로또 예측 시스템 메인 실행 파일 (모든 구현 파일 활용)
작성일: 2025-08-31
버전: 2.0

실행 방법:
    python main.py
    python main.py --predictions 10 --visualize
    python main.py --full-analysis
"""

import sys
import os
import argparse
from pathlib import Path
import json
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 필수 모듈 임포트
try:
    from src.utils.config import Config
    from src.utils.logger import SystemLogger, PredictionLogger, PerformanceLogger
    from src.utils.validator import (
        LottoDataValidator, PredictionValidator, 
        PipelineValidator, ValidationReportGenerator
    )
    from src.data_processing.data_loader import LottoDataLoader
    from src.data_processing.data_cleaner import LottoDataCleaner
    from src.analysis.basic_statistics import BasicStatistics
    from src.analysis.pattern_detector import LottoPatternDetector
    from src.analysis.frequency_analyzer import LottoFrequencyAnalyzer
    from src.analysis.time_series_analyzer import LottoTimeSeriesAnalyzer
    from src.genius_formulas.formula_01_genius_insight import GeniusInsightFormula
    from src.prediction.ensemble_predictor import EnsemblePredictor
    from src.prediction.prediction_synthesizer import PredictionSynthesizer
    from src.prediction.confidence_calculator import ConfidenceCalculator
    
    # 다양성 관리는 ensemble_predictor에 통합되어 있다고 가정
    # from src.prediction.diversity_manager import DiversityGuaranteedPredictor, PredictionHistory
    
except ImportError as e:
    print(f"모듈 임포트 오류: {e}")
    print("필요한 파일들이 모두 구현되었는지 확인해주세요.")
    sys.exit(1)

def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        'data/raw',
        'data/processed', 
        'outputs/predictions',
        'outputs/reports',
        'outputs/visualizations',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_system_setup():
    """시스템 설정 검증"""
    data_path = Path(Config.DATA_FILE_PATH)
    if not data_path.exists():
        print(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        print("lotto_results.csv 파일을 data/raw/ 디렉토리에 위치시켜주세요.")
        return False
    return True

def run_comprehensive_analysis(lotto_data, logger):
    """포괄적 분석 수행"""
    analysis_results = {}
    
    try:
        # 1. 기초 통계 분석
        logger.info("기초 통계 분석 시작")
        stats_analyzer = BasicStatistics(lotto_data)
        stats_results = stats_analyzer.comprehensive_analysis()
        analysis_results['basic_statistics'] = stats_results
        
        # 2. 빈도 분석
        logger.info("빈도 분석 시작")
        freq_analyzer = LottoFrequencyAnalyzer()  # FrequencyAnalyzer → LottoFrequencyAnalyzer
        freq_results = freq_analyzer.analyze_frequencies(lotto_data)  # 메소드명도 확인 필요
        analysis_results['frequency_analysis'] = freq_results
        
        # 3. 패턴 탐지
        logger.info("패턴 탐지 시작")
        pattern_detector = LottoPatternDetector()
        pattern_results = pattern_detector.detect_all_patterns(lotto_data)
        analysis_results['pattern_detection'] = pattern_results
        
        # 4. 시계열 분석
        logger.info("시계열 분석 시작")
        timeseries_analyzer = LottoTimeSeriesAnalyzer()
        timeseries_results = timeseries_analyzer.analyze_time_series(lotto_data)
        analysis_results['time_series'] = timeseries_results
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"포괄적 분석 중 오류: {e}")
        return analysis_results

def run_prediction_pipeline(lotto_data, analysis_results, num_predictions, logger):
   """예측 파이프라인 실행"""
   predictions = []
   
   try:
       # 1. 천재적 통찰 공식 초기화
       logger.info("천재적 통찰 공식 초기화")
       genius_formula = GeniusInsightFormula(lotto_data, analysis_results['basic_statistics'])
       
       # 2. 앙상블 예측기 초기화
       logger.info("앙상블 예측 시스템 초기화")
       ensemble_predictor = EnsemblePredictor(lotto_data, analysis_results['basic_statistics'])
       
       # 3. 예측 합성기 초기화
       logger.info("예측 합성기 초기화")
       prediction_synthesizer = PredictionSynthesizer()
       
       # 4. 신뢰도 계산기 초기화
       logger.info("신뢰도 계산기 초기화")
       confidence_calculator = ConfidenceCalculator()
       
       # 5. 예측 생성
       logger.info(f"{num_predictions}개 예측 생성 시작")
       
       for i in range(num_predictions):
           # 앙상블 예측 수행
           ensemble_result = ensemble_predictor.predict_ensemble()
           
           # 예측 합성
           synthesized_prediction = prediction_synthesizer.synthesize_predictions([ensemble_result])
           
           # 신뢰도 계산 - 수정된 부분
           confidence_result = confidence_calculator.calculate_prediction_confidence(
               synthesized_prediction['final_prediction'],
               {
                   'diversity_score': synthesized_prediction.get('diversity_score', 0.7),
                   'consensus_analysis': {'high_consensus_numbers': []},
                   'prediction_agreement': 0.6,
                   'individual_confidences': [0.6, 0.5, 0.7],
                   'used_methods': synthesized_prediction.get('methods_used', ['ensemble'])
               }
           )
           
           # 신뢰도 값 추출
           confidence = confidence_result['total_confidence']
           
           # 최종 예측 결과
           final_prediction = {
               'prediction_id': i + 1,
               'numbers': synthesized_prediction['final_prediction'],
               'confidence': confidence,
               'methods_used': synthesized_prediction.get('methods_used', []),
               'generation_timestamp': datetime.now().isoformat()
           }
           
           predictions.append(final_prediction)
           
           print(f"예측 {i+1}: {sorted(final_prediction['numbers'])} (신뢰도: {confidence:.3f})")
       
       return predictions
       
   except Exception as e:
       logger.error(f"예측 파이프라인 오류: {e}")
       return predictions

def validate_system_components(lotto_data, config, logger):
    """시스템 컴포넌트 검증"""
    try:
        logger.info("시스템 검증 시작")
        
        # 파이프라인 검증기 초기화
        pipeline_validator = PipelineValidator()
        
        # 모의 컴포넌트 (실제 구현된 것들로 교체)
        components = {
            'data_loader': LottoDataLoader(Config.DATA_FILE_PATH),
            'statistics_analyzer': BasicStatistics(lotto_data),
            'pattern_detector': LottoPatternDetector(),
            'frequency_analyzer': LottoFrequencyAnalyzer(),
            'timeseries_analyzer': LottoTimeSeriesAnalyzer(),
            'predictor': EnsemblePredictor(lotto_data, {})
        }
        
        # 전체 파이프라인 검증
        validation_results = pipeline_validator.validate_complete_pipeline(
            lotto_data, config, components, test_prediction=True
        )
        
        # 검증 리포트 생성
        report_generator = ValidationReportGenerator()
        validation_report = report_generator.generate_comprehensive_report(validation_results)
        
        # 검증 결과 저장
        validation_file = Path('outputs/reports/validation_report.json')
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"검증 리포트 저장: {validation_file}")
        
        return validation_results, validation_report
        
    except Exception as e:
        logger.error(f"시스템 검증 오류: {e}")
        return {}, {}

def generate_comprehensive_report(lotto_data, analysis_results, predictions, validation_report, logger):
    """종합 분석 리포트 생성"""
    try:
        logger.info("종합 리포트 생성 시작")
        
        # HTML 리포트 생성
        html_content = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>로또 예측 시스템 - 종합 분석 리포트</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #667eea; background: #f8f9fa; }}
        .prediction-box {{ background: white; border: 2px solid #667eea; border-radius: 10px; padding: 15px; margin: 10px 0; }}
        .numbers {{ font-size: 24px; font-weight: bold; color: #667eea; text-align: center; }}
        .confidence {{ text-align: center; color: #28a745; font-weight: bold; }}
        .stats-table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .stats-table th {{ background-color: #f2f2f2; }}
        .warning {{ color: #856404; background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; }}
        .success {{ color: #155724; background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>로또 예측 시스템 - 종합 분석 리포트</h1>
        <p>생성 시간: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>1. 데이터 개요</h2>
        <table class="stats-table">
            <tr><th>항목</th><th>값</th></tr>
            <tr><td>총 회차 수</td><td>{len(lotto_data):,}회</td></tr>
            <tr><td>최신 회차</td><td>{lotto_data.iloc[0]['round'] if 'round' in lotto_data.columns else 'N/A'}회</td></tr>
            <tr><td>데이터 기간</td><td>약 {len(lotto_data) * 3.5 / 365:.1f}년</td></tr>
            <tr><td>분석 완료 시간</td><td>{datetime.now().strftime('%H:%M:%S')}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>2. 예측 결과</h2>
"""
        
        # 예측 결과 추가
        for i, prediction in enumerate(predictions, 1):
            numbers = sorted(prediction['numbers'])
            confidence = prediction['confidence']
            
            html_content += f"""
        <div class="prediction-box">
            <h3>예측 세트 {i}</h3>
            <div class="numbers">{' - '.join(map(str, numbers))}</div>
            <div class="confidence">신뢰도: {confidence:.1%}</div>
        </div>
"""
        
        # 시스템 상태 추가
        html_content += f"""
    </div>
    
    <div class="section">
        <h2>3. 시스템 상태</h2>
        <div class="{'success' if validation_report.get('overall_status') == 'excellent' else 'warning'}">
            시스템 상태: {validation_report.get('overall_status', 'unknown').upper()}<br>
            건강 점수: {validation_report.get('system_health_score', 0):.1%}
        </div>
    </div>
    
    <div class="section">
        <h2>4. 분석 요약</h2>
        <ul>
"""
        
        # 분석 결과 요약 추가
        if 'basic_statistics' in analysis_results:
            html_content += "<li>기초 통계 분석 완료</li>"
        if 'frequency_analysis' in analysis_results:
            html_content += "<li>빈도 분석 완료</li>"
        if 'pattern_detection' in analysis_results:
            patterns = analysis_results['pattern_detection']
            total_patterns = sum(len(patterns) for patterns in patterns.values() if isinstance(patterns, dict))
            html_content += f"<li>패턴 탐지 완료: {total_patterns}개 패턴 발견</li>"
        if 'time_series' in analysis_results:
            html_content += "<li>시계열 분석 완료</li>"
        
        html_content += """
        </ul>
    </div>
    
    <div class="section">
        <h2>5. 주의사항</h2>
        <div class="warning">
            <strong>⚠️ 중요한 안내</strong><br>
            • 이 예측은 통계적 분석과 수학적 모델에 기반하지만 로또는 본질적으로 무작위입니다<br>
            • 예측 결과는 참고용으로만 사용하시고, 과도한 투자는 피하세요<br>
            • 도박 중독에 주의하시고 적정한 금액으로 건전하게 즐기세요
        </div>
    </div>
</body>
</html>
"""
        
        # HTML 리포트 저장
        report_file = Path('outputs/reports/analysis_report.html')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"종합 리포트 저장: {report_file}")
        return report_file
        
    except Exception as e:
        logger.error(f"리포트 생성 오류: {e}")
        return None

def save_detailed_results(analysis_results, predictions, logger):
    """상세 결과 저장"""
    try:
        # 1. 예측 결과 저장
        predictions_file = Path('outputs/predictions/latest_prediction.json')
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'summary': {
                'total_predictions': len(predictions),
                'average_confidence': sum(p['confidence'] for p in predictions) / len(predictions) if predictions else 0,
                'unique_predictions': len(set(tuple(sorted(p['numbers'])) for p in predictions))
            }
        }
        
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(prediction_data, f, ensure_ascii=False, indent=2)
        
        # 2. 분석 결과 저장
        analysis_file = Path('outputs/reports/analysis_results.json')
        
        # 분석 결과를 JSON 직렬화 가능한 형태로 변환
        serializable_results = {}
        for key, value in analysis_results.items():
            try:
                if hasattr(value, 'to_dict'):
                    serializable_results[key] = value.to_dict()
                elif isinstance(value, dict):
                    # 중첩된 객체들도 변환
                    serializable_dict = {}
                    for k, v in value.items():
                        if hasattr(v, 'to_dict'):
                            serializable_dict[k] = v.to_dict()
                        elif isinstance(v, (list, dict, str, int, float, bool)) or v is None:
                            serializable_dict[k] = v
                        else:
                            serializable_dict[k] = str(v)
                    serializable_results[key] = serializable_dict
                else:
                    serializable_results[key] = str(value)
            except Exception:
                serializable_results[key] = f"직렬화 불가: {type(value)}"
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 3. 히스토리 업데이트 - 시간 정보 추가
        history_file = Path('outputs/predictions/prediction_history.txt')
        if not history_file.exists():
            with open(history_file, 'w', encoding='utf-8') as f:
                # 헤더 작성 - 시간 컬럼 추가
                f.write("날짜 | 시간 | ID | 예측번호 | 신뢰도\n")

        with open(history_file, 'a', encoding='utf-8') as f:
            current_time = datetime.now()
            date_str = current_time.strftime('%Y-%m-%d')
            time_str = current_time.strftime('%H:%M:%S')
            
            for prediction in predictions:
                # 번호를 쉼표와 공백으로 구분
                numbers_str = ', '.join(map(str, sorted(prediction['numbers'])))
                confidence_str = f"{prediction['confidence']*100:.2f}%"
                
                # 파이프로 구분된 테이블 형태 - 시간 컬럼 추가
                f.write(f"{date_str} | {time_str} | {prediction['prediction_id']} | {numbers_str} | {confidence_str}\n")
        
        logger.info("모든 결과 파일 저장 완료")
        
    except Exception as e:
        logger.error(f"결과 저장 오류: {e}")

def main():
    """메인 실행 함수"""
    print("🎯 로또 예측 시스템 시작 (Enhanced)")
    print("=" * 60)
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="천재적 사고 기반 로또 예측 시스템")
    parser.add_argument('--predictions', '-p', type=int, default=5,
                       help='생성할 예측 세트 수 (기본값: 5)')
    parser.add_argument('--full-analysis', '-f', action='store_true',
                       help='전체 고급 분석 수행')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='결과 시각화 활성화')
    parser.add_argument('--validate', action='store_true',
                       help='시스템 검증 수행')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='로그 레벨 설정')
    
    args = parser.parse_args()
    
    try:
        # 1. 환경 설정
        print("1️⃣ 환경 설정")
        create_directories()
        
        # 로거 초기화
        system_logger = SystemLogger("LottoPredictionSystem")
        logger = system_logger.get_logger("main")
        system_logger.log_system_info()
        logger.info("로또 예측 시스템 시작")

        # 예측 및 성능 로거도 초기화
        prediction_logger = PredictionLogger(system_logger)
        performance_logger = PerformanceLogger(system_logger)
        
        if not validate_system_setup():
            return 1
        
        # 2. 데이터 로딩 및 전처리
        print("\n2️⃣ 데이터 로딩 및 전처리")
        logger.info("데이터 로딩 시작")
        
        data_loader = LottoDataLoader(Config.DATA_FILE_PATH)
        raw_data = data_loader.load_and_preprocess()
        
        # 데이터 정제
        data_cleaner = LottoDataCleaner()
        cleaned_data = data_cleaner.clean_data(raw_data)
        
        print(f"✅ {len(cleaned_data)} 회차 데이터 로딩 완료")
        
        # 3. 데이터 검증
        if args.validate:
            print("\n3️⃣ 데이터 검증")
            data_validator = LottoDataValidator()
            validation_result = data_validator.validate_dataframe(cleaned_data)
            print(f"데이터 검증: {validation_result.get_summary()}")
            
            if not validation_result.is_valid:
                print("⚠️ 데이터 검증 실패. 계속 진행하지만 결과에 영향을 줄 수 있습니다.")
                for error in validation_result.errors:
                    print(f"   오류: {error}")
        
        # 4. 분석 수행
        if args.full_analysis:
            print("\n4️⃣ 포괄적 분석 수행")
            analysis_results = run_comprehensive_analysis(cleaned_data, logger)
        else:
            print("\n4️⃣ 기본 분석 수행")
            # 기본 분석만 수행
            stats_analyzer = BasicStatistics(cleaned_data)
            basic_stats = stats_analyzer.comprehensive_analysis()
            analysis_results = {'basic_statistics': basic_stats}
        
        print(f"✅ 분석 완료: {len(analysis_results)}개 분석 모듈")
        
        # 5. 예측 생성
        print(f"\n5️⃣ {args.predictions}개 예측 생성")
        predictions = run_prediction_pipeline(
            cleaned_data, analysis_results, args.predictions, logger
        )
        
        if not predictions:
            print("❌ 예측 생성 실패")
            return 1
        
        print(f"✅ {len(predictions)}개 예측 생성 완료")
        
        # 6. 시스템 검증 (옵션)
        validation_results = {}
        validation_report = {}
        if args.validate:
            print("\n6️⃣ 시스템 검증")
            config_dict = {
                'data_file_path': Config.DATA_FILE_PATH,
                'prediction_count': args.predictions,
                'confidence_threshold': 0.1
            }
            validation_results, validation_report = validate_system_components(
                cleaned_data, config_dict, logger
            )
            print(f"시스템 상태: {validation_report.get('overall_status', 'unknown')}")
        
        # 7. 결과 저장
        print("\n7️⃣ 결과 저장")
        save_detailed_results(analysis_results, predictions, logger)
        
        # 8. 종합 리포트 생성
        print("\n8️⃣ 종합 리포트 생성")
        report_file = generate_comprehensive_report(
            cleaned_data, analysis_results, predictions, validation_report, logger
        )
        
        if report_file:
            print(f"✅ 종합 리포트: {report_file}")
            print(f"🌐 브라우저에서 확인: file://{report_file.absolute()}")
        
        # 9. 시각화 (옵션)
        if args.visualize:
            print("\n9️⃣ 결과 시각화")
            try:
                from src.utils.visualizer import LottoVisualizer  # 여기서 임포트
                
                visualizer = LottoVisualizer()
                charts = visualizer.create_comprehensive_report(
                    cleaned_data, analysis_results, predictions
                )
                print(f"시각화 완료: visualizations 폴더에 차트 생성")
                
            except Exception as e:
                logger.warning(f"시각화 오류: {e}")
                print(f"⚠️ 시각화 중 오류: {e}")
        
        # 10. 최종 요약
        print(f"\n🎉 예측 시스템 실행 완료!")
        print("=" * 60)
        print("📊 생성된 결과:")
        print(f"   • 예측 세트: {len(predictions)}개")
        print(f"   • 평균 신뢰도: {sum(p['confidence'] for p in predictions) / len(predictions):.1%}")
        print(f"   • 고유 예측: {len(set(tuple(sorted(p['numbers'])) for p in predictions))}개")
        
        print("\n📁 저장된 파일:")
        print(f"   • outputs/predictions/latest_prediction.json")
        print(f"   • outputs/predictions/prediction_history.csv") 
        print(f"   • outputs/reports/analysis_report.html")
        if args.full_analysis:
            print(f"   • outputs/reports/analysis_results.json")
        if args.validate:
            print(f"   • outputs/reports/validation_report.json")
        
        print(f"\n📋 로그 파일: logs/lotto_system.log")
        
        print("\n" + "=" * 60)
        print("💡 이 예측은 통계적 분석과 창의적 사고를 기반으로 하지만")
        print("   로또는 본질적으로 무작위이므로 적당한 금액으로 즐기세요!")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 사용자에 의해 중단되었습니다.")
        return 1
        
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류 발생: {e}")
        print("디버깅을 위한 상세 오류:")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
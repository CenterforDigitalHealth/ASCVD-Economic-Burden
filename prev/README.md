# ASCVD-Economic-Burden
ASCVD-Economic-Burden

## 실행 방법

### 1. 시뮬레이션 실행
```bash
bash run.sh
```
- 여러 시나리오를 병렬로 실행합니다
- 결과는 `tmpresults/` 디렉토리에 저장됩니다
- 진행 상황은 `logs/` 디렉토리에서 확인할 수 있습니다

### 2. 결과 합치기
```bash
python combine.py
```
- `tmpresults/`의 개별 결과 파일들을 합쳐서 `results/aggregate_results.csv`를 생성합니다

### 3. Imputation (결과 보정)
```bash
python imputation.py -i results/aggregate_results.csv -o results/aggregate_results_imputed.csv
```
- 음수 값이나 누락된 값을 통계적 모델로 보정합니다
- 최종 결과는 `results/` 디렉토리에 저장됩니다

### 4. 테이블 생성
```bash
python generate_tables.py -f results/aggregate_results_imputed.csv -d 0.02 -i 0.11
```
- `-d`: discount rate (0, 0.02, 0.03 등)
- `-i`: informal care rate (0, 0.05, 0.11, 0.23 등)
- 결과는 `tables/` 디렉토리에 저장됩니다

### 전체 파이프라인 (한 번에 실행)
```bash
# 기존 결과 삭제 (선택사항)
rm -f results/aggregate_results.csv
rm -f results/aggregate_results_imputed.csv
rm -f tmpresults/aggregate_results.csv

# 전체 파이프라인 실행
bash run.sh && \
python combine.py && \
python imputation.py -i results/aggregate_results.csv -o results/aggregate_results_imputed.csv && \
python generate_tables.py -f results/aggregate_results_imputed.csv -d 0.02 -i 0.11
```

## 주요 변경 사항 (Dieleman Cost)
- `HMM_main.py`는 이제 기본적으로 IDF 데이터 대신 **Dieleman et al. (2020)** 기반의 치료비용(`TC_dieleman.csv`)을 사용합니다.
- 이는 [Nat Med] 논문의 결과와 더 일치하는 경제적 부담 추정치를 생성합니다.

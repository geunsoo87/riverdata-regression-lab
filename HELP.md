# RiverData Regression Lab Help

## 1) 앱 개요
- 이 앱은 CSV/XLSX 데이터를 업로드한 뒤, 패널별로 X/Y를 선택하여 회귀 그래프를 만들고 영향력 진단(이상치 후보)을 확인하는 도구입니다.
- 각 패널은 독립적으로 설정할 수 있으며, OLS/Robust 회귀, CI/PI 표시 여부, 축 범위, 라벨, 내보내기를 지원합니다.

## 2) 빠른 사용 순서
1. `Upload CSV or XLSX`로 데이터를 업로드합니다.
2. 상단 `Settings / Panel Selection`에서 패널 수, X/Y, 축/라벨, 표시 옵션을 설정합니다.
3. 그래프와 `Outlier Diagnostics`를 확인합니다.
4. 필요하면 `Apply outlier removal`을 켜서 제거 전/후 비교를 확인합니다.
5. `Export`에서 PNG/PDF/SVG로 저장합니다.

## 3) 주요 설정 의미
- `Use robust regression line (RLM)`:
  이상치 영향이 큰 경우 robust 선을 사용합니다. (구간 CI/PI는 OLS 기반)
- `Show 95% CI`:
  평균 반응의 95% 신뢰구간 음영을 표시합니다.
- `Show 95% PI`:
  개별 관측값의 95% 예측구간 음영을 표시합니다.
- `Show legend`:
  범례 표시/숨김.
- `Apply outlier removal`:
  `is_outlier_residual=True` 행만 제외하고 회귀를 다시 계산합니다.

## 4) 이상치 후보 기준 (자동 판정)
- `is_outlier_residual = outside_95_pi OR (|rstudent| > 3)`를 사용합니다.
- `is_influential = (Cook's D > 4/n) OR (leverage > 2*p/n)`를 사용합니다. (단순회귀 p=2)
- Cook's D/leverage는 영향점 진단용이며, 이상치 자동 제거 기준으로 사용하지 않습니다.
- 여기서 `n`은 해당 패널에서 유효한 분석 행 수입니다.
- `leverage`는 영향력 판단 보조 지표로 표에 제공합니다.

## 5) 진단 표 컬럼 해석
- `rstudent`:
  외부 studentized residual. 절댓값이 클수록 모델에서 벗어난 관측일 가능성이 큼.
- `cooks_d`:
  해당 점이 회귀계수에 미치는 영향력.
- `leverage`:
  X 공간에서의 영향도(극단 X 여부).
- `is_outlier_residual`:
  잔차 기반 이상치 후보 여부 (`outside_95_pi` 또는 `|rstudent|>3`).
- `is_influential`:
  영향점 여부 (`Cook's D` 또는 `leverage` 임계치 초과).
- `outlier_class`:
  `BOTH`, `RESIDUAL_OUTLIER`, `INFLUENTIAL`, `NONE` 중 하나.
- `action`:
  `QC_CHECK`, `SENSITIVITY`, `PRIORITY_REVIEW`, `OK`, `MANUAL_REVIEW` 권장 라벨.
- `outside_95_ci`:
  관측값이 95% CI(평균 반응 구간) 밖인지 여부.
- `outside_95_pi`:
  관측값이 95% PI(개별 관측 구간) 밖인지 여부.
- `manual_review`:
  분석자가 수동으로 검토 대상 표시(체크박스).

## 6) outside_95_ci / outside_95_pi 해석 팁
- `outside_95_ci=True`:
  평균 반응 추정 구간 기준으로도 벗어남. 평균 추세 대비 이탈 강함.
- `outside_95_pi=True`:
  개별 관측 허용 범위를 벗어남. 관측치 관점에서 이례적일 가능성.
- 일반적으로 PI가 CI보다 넓으므로, `outside_95_ci=True`이고 `outside_95_pi=False`인 경우도 가능합니다.

## 7) manual_review 활용 권장
- 자동 판정과 별도로, 도메인 지식으로 수동 체크를 남기세요.
- 예:
  - 측정 오류 의심
  - 홍수/갈수 등 특수 이벤트 구간
  - 센서 교체 전후의 체계적 편차
- 현재 manual_review는 검토 기록용이며, 자동 제거와는 분리됩니다.

## 8) Preset(설정 저장/불러오기)
- `Preset (Save current settings)`:
  현재 패널/스타일/라벨 설정을 JSON으로 저장.
- `Preset (Load saved settings)`:
  저장한 JSON을 다시 불러와 설정을 복원.
- 동일한 컬럼 구조를 가진 파일에 매우 빠르게 재적용할 수 있습니다.

## 9) 보고/논문 작성 시 권장 워크플로우
1. 자동 후보 확인 (`is_outlier_residual`, `is_influential`, `outlier_class`)
2. CI/PI outside 여부 함께 확인
3. `manual_review`로 근거 기록
4. 제거 전/후 계수/R2/RMSE 비교
5. 최종 포함/제외 기준을 Methods에 명시

## 10) 주의 사항
- 표의 후보 판정은 통계적 힌트이며, 자동으로 진실/오류를 확정하지 않습니다.
- 데이터 맥락(현장 상황, 측정 조건, 이벤트 정보)과 함께 해석하세요.
## 11) X/Y log10 조합 사용
- 각 패널에서 `Apply log10 to X`, `Apply log10 to Y`를 독립적으로 켤 수 있습니다.
- 조합 예시:
  - 둘 다 OFF: 선형-선형
  - X ON, Y OFF: 로그-선형
  - X OFF, Y ON: 선형-로그
  - 둘 다 ON: 로그-로그
- log10 적용 시 해당 축 값이 `<= 0`인 행은 분석에서 제외되고 경고가 표시됩니다.
- 수동 축 범위(manual range)는 변환된 값(로그값) 기준으로 입력합니다.

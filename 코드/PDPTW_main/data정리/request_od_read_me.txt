request_od.py 상세 설명서
===========================

대상 파일
---------

코드/PDPTW_main/data정리/request_od.py


1. 프로그램의 목적
------------------

이 스크립트는 CSV에 들어 있는 모든 요청의 time window를 일정한 시간 간격으로
나눈 뒤, 각 Origin → Destination(이하 OD) 조합에서 시간대별로 몇 개의 요청
시간창이 활성화되는지를 히트맵 PNG로 저장한다.

히트맵의 각 축과 셀은 다음 의미를 갖는다.

* 세로축의 한 행: 하나의 방향성 OD 조합
* 가로축의 한 열: 하나의 시간 bin
* 셀의 값과 색: 해당 OD에서 해당 시간 bin과 겹치는 요청 time window의 수

예를 들어 다음 요청이 있다고 가정한다.

    요청 A: 1 → 2, 06:15~06:20
    요청 B: 1 → 2, 06:17~06:22
    요청 C: 2 → 1, 06:18~06:23

5분 bin을 사용하면 개념적으로 다음 행렬이 만들어진다.

    OD       06:15~06:20    06:20~06:25
    1 → 2         2               1
    2 → 1         1               1

한 요청이 둘 이상의 bin과 조금이라도 겹치면 각 bin에 1씩 더한다. 겹친 시간의
길이를 가중치로 사용하지 않으므로, 어떤 bin과 1분만 겹치든 5분 전체가 겹치든
그 셀에는 동일하게 1이 더해진다.

기본 입력 CSV를 기준으로 확인된 결과는 다음과 같다.

* 전체 요청: 3,320건
* 방향성이 구분된 OD 조합: 75개
* 기본 시간 간격: 5분
* 시간축: 06:15부터 19:20까지 158개 bin
* 한 셀의 최대 활성 요청 수: 10개
* 전체 행렬의 집계값 합: 6,363
* 값이 1 이상인 셀 수: 3,285
* 기본 출력 파일명: rolling_horizon_request_status_all_time_windows_od.png

행렬의 집계값 합이 요청 수보다 큰 이유는 한 요청의 time window가 여러 시간
bin에 걸칠 수 있기 때문이다.

final_status 열은 사용하지 않으므로 완료, 지연 등의 상태와 관계없이 CSV의 모든
행을 집계한다. selected_request_id를 이용한 중복 제거도 하지 않으므로, 동일한
요청이 여러 행에 있으면 각 행이 별도의 요청으로 집계된다.


2. 전체 실행 흐름
-----------------

    스크립트 실행
        ↓
    main()
        ├─ parse_args()
        ├─ 명령행 인수 검증
        ├─ 입력 CSV 확인 및 읽기
        ├─ 필수 열 검증
        ├─ prepare_od_time_windows()
        │    ├─ hhmm_to_minutes()
        │    └─ sortable_node()
        ├─ save_heatmap()
        └─ 실행 결과 출력
             ↓
           PNG 저장


3. 모듈 설명문: 1~6행
---------------------

파일 맨 위의 삼중 따옴표 문자열은 모듈 docstring이다. 스크립트가 전체 요청의
time window를 시간대별, OD별 히트맵으로 만든다는 목적을 설명한다.

여기서 시간창은 기본적으로 [start, end)로 해석된다.

* 시작 시각 start는 포함한다.
* 종료 시각 end는 포함하지 않는다.

따라서 06:15~06:20 요청은 06:15~06:20 bin에는 포함되지만, 정확히 06:20에
시작하는 다음 bin에는 중복 집계되지 않는다.


4. import 및 Matplotlib 백엔드: 8~19행
-----------------------------------------

4.1 from __future__ import annotations

함수 타입 힌트를 즉시 실제 객체로 평가하지 않고 지연된 형태로 보관한다.
타입 힌트는 실행 결과를 바꾸는 로직이 아니라 코드 작성자, IDE, 정적 타입 검사
도구가 매개변수와 반환값의 형식을 이해하도록 돕는다.

4.2 argparse

터미널에서 --input, --output, --time-bin, --dpi 같은 명령행 옵션을 받는 데
사용한다.

4.3 pathlib.Path

문자열 대신 경로 객체를 사용한다. 절대 경로 변환, 파일 존재 여부 확인, 상위
폴더 생성, 파일명과 확장자 변경, 운영체제별 경로 처리에 쓰인다.

4.4 matplotlib

OD × 시간 행렬을 히트맵으로 그리고 PNG로 저장한다.

4.5 numpy as np

다음 작업에 사용한다.

* 2차원 정수 집계 행렬 생성
* 시간 bin 및 tick 위치 배열 생성
* 종료 시각의 bin 경계 올림 계산
* 이산 색상의 경계 배열 생성

4.6 pandas as pd

다음 작업에 사용한다.

* CSV 읽기
* origin과 destination 문자열 정리
* HH:MM 문자열 파싱
* OD × 시간 집계 결과를 DataFrame으로 표현

4.7 matplotlib.use("Agg")

Agg는 GUI 창 없이 그림을 렌더링하는 Matplotlib 백엔드다. 이 스크립트를
실행해도 그래프 창은 뜨지 않고 fig.savefig()를 통해 PNG 파일만 생성된다.
따라서 GUI가 없는 서버나 원격 터미널에서도 실행할 수 있다.

백엔드는 pyplot을 import하기 전에 선택해야 하며 현재 코드의 순서가 올바르다.

4.8 matplotlib.pyplot

figure, axes, 히트맵, 라벨, 제목, colorbar를 만들고 PNG로 저장하는 고수준
인터페이스다.

4.9 BoundaryNorm

셀 값이 정수이므로 연속 실수 범위가 아니라 1, 2, 3 등의 정수 단계별로 색을
구분하는 데 사용한다. 값 0은 별도로 흰색으로 표시한다.


5. 전역 상수: 22~40행
---------------------

5.1 PDPTW_ROOT

    PDPTW_ROOT = Path(__file__).resolve().parents[3]

__file__은 현재 request_od.py 파일의 경로다. resolve()는 이를 절대 경로로
바꾼다. 현재 폴더 구조에서 parents 값은 다음과 같다.

    parents[0] = data정리
    parents[1] = PDPTW_main
    parents[2] = 코드
    parents[3] = PDPTW_Rolling_Horizon

따라서 PDPTW_ROOT는 저장소 최상위 폴더를 가리킨다. 이 방식은 터미널의 현재
작업 폴더와 관계없이 기본 입력 경로를 request_od.py 위치 기준으로 찾게 해준다.

단, request_od.py를 다른 깊이의 폴더로 옮기면 parents[3]가 더 이상 저장소
루트를 가리키지 않을 수 있다.

5.2 DEFAULT_INPUT

기본 입력 파일은 다음 위치다.

    자료/결과/Ortools/d5000/Penalty_Per_Vehicles/Original/130000/
    vehicles_150/rolling_horizon_request_status.csv

Path 객체에 / 연산자를 사용하여 하위 경로를 단계적으로 연결한다. 사용자가
--input을 지정하지 않으면 이 CSV를 사용한다.

5.3 REQUIRED_COLUMNS

    origin
    destination
    time_window_start_hhmm
    time_window_end_hhmm

스크립트가 반드시 필요로 하는 네 열이다.

* origin: 요청의 출발 노드
* destination: 요청의 도착 노드
* time_window_start_hhmm: time window 시작 시각
* time_window_end_hhmm: time window 종료 시각

selected_request_id, assigned_vehicle, pickup_time, final_status 등 나머지 열은
사용하지 않는다.

REQUIRED_COLUMNS는 list가 아니라 set이다. set은 열 순서를 보장하기 위한
자료구조가 아니지만, 이후 코드가 열의 위치가 아닌 이름으로 접근하므로 현재
집계 결과에는 영향을 주지 않는다.


6. parse_args(): 43~61행
-------------------------

정의:

    def parse_args() -> argparse.Namespace:

역할:

터미널에서 전달된 명령행 옵션을 읽어 argparse.Namespace로 반환한다. 이
함수는 CSV를 읽거나 그림을 만들지 않고 실행 설정만 해석한다.

6.1 ArgumentParser 생성

프로그램 설명이 포함된 명령행 파서를 만든다. 다음 명령을 실행하면 설명과
옵션을 확인할 수 있다.

    python ".\코드\PDPTW_main\data정리\request_od.py" --help

6.2 --input

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)

입력 CSV 경로다.

* 자료형: Path
* 생략 시: DEFAULT_INPUT

argparse는 입력 문자열을 Path로 바꾸지만 이 시점에는 파일이 존재하는지
확인하지 않는다. 실제 파일 확인은 main()이 담당한다.

6.3 --output

출력 PNG 경로다.

* 자료형: Path
* 생략 시: None

None이면 main()이 입력 CSV와 같은 폴더에 다음 규칙으로 이름을 만든다.

    입력:  requests.csv
    출력:  requests_all_time_windows_od.png

6.4 --time-bin

시간축을 몇 분 간격으로 나눌지 지정한다.

* 자료형: int
* 기본값: 5
* 허용 조건: 양수이고 60의 약수

main()의 검증을 통과하는 값은 다음과 같다.

    1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60

간격이 작을수록 시간 해상도는 높아지지만 행렬의 열 수와 이미지 너비가
커진다.

6.5 --dpi

출력 이미지의 해상도다.

* 자료형: int
* 기본값: 300
* 허용 조건: 양수

DPI가 높을수록 글자와 선이 선명해지는 대신 이미지 파일 크기와 렌더링 부담이
커진다.

6.6 반환

    return parser.parse_args()

예를 들어 다음과 같이 실행하면:

    python request_od.py --input sample.csv --time-bin 10 --dpi 200

개념적으로 다음 객체를 반환한다.

    Namespace(
        input=Path("sample.csv"),
        output=None,
        time_bin=10,
        dpi=200,
    )


7. hhmm_to_minutes(): 64~71행
--------------------------------

정의:

    def hhmm_to_minutes(series: pd.Series, column_name: str) -> pd.Series:

역할:

"06:15" 같은 HH:MM 문자열을 자정 이후 누적 분으로 변환한다.

    00:00 → 0
    01:30 → 90
    06:15 → 375
    12:00 → 720
    19:20 → 1160
    23:59 → 1439

7.1 매개변수

series는 변환할 시간 열이다. column_name은 변환 오류 메시지에 표시할 열
이름이다. column_name은 계산에는 쓰이지 않고 어느 열의 값이 잘못되었는지
알려주는 용도로 사용된다.

7.2 문자열 정리

    cleaned = series.astype("string").str.strip()

Pandas 문자열 dtype으로 바꾸고 앞뒤 공백을 제거한다.

    " 06:15 " → "06:15"

결측값은 Pandas 문자열 결측값인 <NA>로 유지된다.

7.3 HH:MM 파싱

    parsed = pd.to_datetime(cleaned, format="%H:%M", errors="coerce")

%H는 24시간제 시, %M은 분이다. errors="coerce"이므로 변환할 수 없는
값은 즉시 예외를 일으키는 대신 NaT가 된다.

    "06:15" → 정상 datetime
    "18:40" → 정상 datetime
    "abc"   → NaT
    "25:00" → NaT

7.4 잘못된 비결측 문자열 찾기

    invalid = cleaned.notna() & parsed.isna()

원래 값은 존재하지만 HH:MM으로 파싱하지 못한 행을 찾는다. 원래부터 비어 있던
값은 여기서 잘못된 문자열로 보고하지 않고, prepare_od_time_windows()에서
결측 행으로 검출한다.

7.5 오류 예시 추출

잘못된 값이 있으면 중복을 제거한 후 최대 5개를 Python list로 만든다.

    cleaned.loc[invalid].drop_duplicates().head(5).tolist()

예를 들어 다음과 같은 목록을 만들 수 있다.

    ["25:30", "오전 6시", "6시 10분"]

7.6 오류 발생

오류 메시지 예시는 다음과 같다.

    ValueError: time_window_start_hhmm에 잘못된 HH:MM 값이 있습니다:
    ['25:30', 'abc']

7.7 누적 분 반환

    return parsed.dt.hour * 60 + parsed.dt.minute

계산식은 다음과 같다.

    누적 분 = 시 × 60 + 분

반환값은 입력 Series와 같은 인덱스를 가진 Pandas Series다.

7.8 제한

날짜는 처리하지 않는다. 하루 안의 HH:MM만 다루므로 23:55 → 00:10 같은
자정 통과 시간창은 이후 종료가 시작보다 이른 것으로 판정된다.


8. sortable_node(): 74~79행
-----------------------------

정의:

    def sortable_node(value: str) -> tuple[int, float | str]:

역할:

노드 이름을 사람이 기대하는 순서로 정렬할 키를 만든다. 숫자 문자열을 그대로
정렬하면 사전식 순서 때문에 다음처럼 된다.

    "1", "10", "11", "2", "20", "3"

이 함수는 숫자로 해석 가능한 노드를 숫자로 바꿔 다음처럼 정렬되게 한다.

    "1", "2", "3", "10", "11", "20"

8.1 문자열 변환과 공백 제거

    text = str(value).strip()

입력값을 문자열로 만들고 앞뒤 공백을 제거한다.

8.2 숫자 변환 성공

    return 0, float(text)

숫자로 바꿀 수 있으면 (0, 숫자값)을 반환한다.

    sortable_node("2")   → (0, 2.0)
    sortable_node("10")  → (0, 10.0)
    sortable_node("3.5") → (0, 3.5)

첫 번째 원소 0은 숫자 노드를 문자 노드보다 먼저 배치하는 그룹 번호다.

8.3 숫자 변환 실패

    except ValueError:
        return 1, text

숫자로 바꿀 수 없으면 (1, 문자열)을 반환한다.

    sortable_node("Depot") → (1, "Depot")
    sortable_node("A")     → (1, "A")

Python 튜플은 첫 번째 원소부터 비교한다. 따라서 숫자 노드 그룹 0이 문자 노드
그룹 1보다 먼저 오며, float와 str을 직접 비교하는 문제도 피한다.

8.4 주의점

"1", "01", "1.0"은 OD 식별 시 서로 다른 문자열이므로 별도의 OD가 될 수
있다. 이 함수는 값을 정규화하는 것이 아니라 정렬 기준만 숫자로 바꾼다.


9. prepare_od_time_windows(): 82~129행
----------------------------------------

정의:

    def prepare_od_time_windows(data: pd.DataFrame, time_bin: int) -> pd.DataFrame:

역할:

이 함수가 핵심 집계를 담당한다. CSV 데이터를 받아 행이 OD, 열이 시간 bin,
값이 활성 요청 수인 DataFrame을 반환한다.

예시 반환 형태:

             375  380  385
    1 → 2      2    1    0
    1 → 3      0    1    1
    2 → 1      1    1    0

열 이름은 HH:MM이 아니라 자정 이후 누적 분이다.

    375 = 06:15
    380 = 06:20
    385 = 06:25

9.1 필요한 열만 복사

    working = data[list(REQUIRED_COLUMNS)].copy()

필요한 네 열만 선택해 복사한다. 이후 working을 수정해도 원본 data는 바뀌지
않는다.

이 함수를 main()을 거치지 않고 직접 호출했는데 필수 열이 없으면 이 줄에서
Pandas KeyError가 난다. 정상 CLI 실행에서는 main()이 먼저 필수 열을 검사한다.

9.2 OD 문자열 정리

origin과 destination을 Pandas 문자열 dtype으로 바꾸고 앞뒤 공백을 제거한다.

    origin=" 6 " → "6"

9.3 시작 및 종료 시각 변환

hhmm_to_minutes()를 호출하여 새 start 및 end 열을 만든다.

    06:15 → 375
    06:22 → 382

9.4 결측 또는 빈 값 검증

다음 조건 중 하나라도 참인 행을 invalid로 지정한다.

* origin이 결측
* origin이 빈 문자열
* destination이 결측
* destination이 빈 문자열
* start가 결측
* end가 결측

잘못된 행이 있으면 처음 10개의 DataFrame 인덱스를 오류 메시지에 표시한다.

    ValueError: OD 또는 time window가 잘못된 행이 있습니다: [10, 23, 51]

pd.read_csv()의 기본 인덱스는 0부터 시작하므로, 헤더를 포함해 CSV 편집기에서
보이는 실제 줄 번호와 차이가 날 수 있다.

9.5 종료가 시작보다 빠른 경우 검증

    if (working["end"] < working["start"]).any():

18:00 → 17:30 같은 요청을 거부한다. 23:55 → 00:10 같은 자정 통과 시간창도
같은 이유로 거부된다.

종료와 시작이 같은 길이 0의 요청은 end < start가 아니므로 허용된다.

9.6 고유 OD 조합 생성

    od_pairs = sorted(
        set(zip(working["origin"], working["destination"])),
        key=lambda pair: (sortable_node(pair[0]), sortable_node(pair[1])),
    )

zip은 각 행의 origin과 destination을 튜플로 묶고, set은 중복 OD를 제거한다.
sorted는 먼저 origin을 정렬하고 origin이 같으면 destination을 정렬한다.

    ("1", "2")와 ("2", "1")은 서로 다른 OD다.

sortable_node()를 사용하므로 1 → 10이 1 → 2보다 앞서는 문자열 정렬 문제를
피한다. 기본 데이터의 앞부분은 다음과 같은 순서가 된다.

    1 → 2
    1 → 3
    1 → 5
    1 → 6
    1 → 7
    1 → 8
    1 → 9
    1 → 10
    2 → 1
    2 → 4

9.7 전체 첫 bin 계산

    first_bin = int(working["start"].min() // time_bin * time_bin)

모든 요청 중 가장 이른 시작 시각을 가져와 그 시각이 포함되는 bin의 시작점으로
내린다.

5분 bin에서 06:22는 382분이다.

    382 // 5 × 5 = 380 = 06:20

따라서 06:22 요청은 06:20~06:25 bin부터 집계된다. 기본 데이터의 first_bin은
375, 즉 06:15다.

9.8 전체 마지막 경계 계산

    last_edge = int(np.ceil(working["end"].max() / time_bin) * time_bin)

모든 요청 중 가장 늦은 종료 시각을 bin 경계로 올린다. 예를 들어 5분 bin에서
가장 늦은 종료가 19:22이면 last_edge는 19:25가 된다.

last_edge는 마지막 열 자체가 아니라 마지막 구간의 오른쪽 경계다.

9.9 시간 bin 목록 생성

    time_bins = list(range(first_bin, last_edge, time_bin))

range의 끝값은 포함되지 않는다. 예를 들어 first_bin=375, last_edge=390,
time_bin=5라면 다음 목록이 된다.

    [375, 380, 385]

각 숫자는 다음 구간의 시작을 의미한다.

    375 → 06:15~06:20
    380 → 06:20~06:25
    385 → 06:25~06:30

기본 데이터에서는 158개 열이 생성된다.

    첫 열:   375  = 06:15~06:20
    마지막: 1160 = 19:20~19:25

9.10 OD 행 위치 사전

    od_index = {pair: index for index, pair in enumerate(od_pairs)}

각 OD가 행렬의 몇 번째 행인지 빠르게 찾기 위한 dict다.

    {
        ("1", "2"): 0,
        ("1", "3"): 1,
        ("2", "1"): 2,
    }

9.11 시간 열 위치 사전

    bin_index = {minute: index for index, minute in enumerate(time_bins)}

각 누적 분이 행렬의 몇 번째 열인지 알려준다.

    {375: 0, 380: 1, 385: 2}

두 dict 덕분에 매 요청마다 list 전체를 검색하지 않고 행과 열 위치를 바로 찾을
수 있다.

9.12 0으로 채운 행렬 생성

    values = np.zeros((len(od_pairs), len(time_bins)), dtype=int)

행 수는 OD 개수, 열 수는 시간 bin 개수인 2차원 정수 배열을 만든다. 기본
데이터에서는 shape가 (75, 158)이고 처음에는 모든 값이 0이다.

9.13 요청 순회

    for row in working.itertuples(index=False):

모든 요청을 한 행씩 순회한다. itertuples()는 각 행을 튜플 형태로 가져오며,
일반적으로 iterrows()보다 빠르다. index=False이므로 DataFrame 인덱스는
튜플에 포함하지 않는다.

9.14 현재 요청의 첫 bin

    first = int(row.start // time_bin * time_bin)

요청 시작 시각을 포함하는 bin 시작점으로 내린다.

    시작 06:22 → 첫 bin 06:20

9.15 현재 요청의 종료 경계

    final = int(np.ceil(row.end / time_bin) * time_bin)

종료 시각을 bin 경계로 올린다.

    종료 06:27 → final 06:30

이후 range(first, final, time_bin)을 사용하므로 final 자체는 포함되지 않는다.

9.16 [start, end) 집계

06:15~06:20 요청은 06:15~06:20 bin에만 포함되고, 종료 시각 06:20을 다음
bin에 중복 집계하지 않는다.

9.17 시작과 종료가 같은 요청

    if final == first:
        final += time_bin

06:20 → 06:20처럼 bin 경계에서 시작과 종료가 같은 요청은 그대로 두면
range(380, 380, 5)가 비어 아무 셀에도 표시되지 않는다. 그래서 final에
time_bin을 더하여 최소 한 bin에 표시한다.

엄밀한 [start, end) 구간에서는 길이 0인 시간창은 어떤 bin과도 겹치지 않지만,
이 스크립트는 시각화에서 요청이 사라지지 않도록 의도적으로 한 bin에 표시한다.

9.18 OD 행 찾기

    od_row = od_index[(row.origin, row.destination)]

현재 요청의 OD가 집계 행렬에서 어느 행인지 가져온다.

9.19 겹치는 모든 bin에 누적

    for minute in range(first, final, time_bin):
        values[od_row, bin_index[minute]] += 1

현재 요청이 겹치는 모든 시간 bin을 순회해 각 셀에 1씩 더한다.

예를 들어 다음 요청을 생각한다.

    OD: 4 → 9
    time window: 06:22~06:27
    time_bin: 5분

계산 결과는 다음과 같다.

    first = 06:20
    final = 06:30
    순회 bin = 06:20, 06:25

따라서 다음 두 셀에 각각 1을 더한다.

    (4 → 9, 06:20~06:25) += 1
    (4 → 9, 06:25~06:30) += 1

요청 시간창 자체는 5분이지만 bin 경계와 정렬되지 않았기 때문에 두 bin에
집계된다.

9.20 표시용 OD 라벨 생성

    labels = [f"{origin} → {destination}" for origin, destination in od_pairs]

튜플 형태의 OD를 표시 문자열로 바꾼다.

    ("1", "2") → "1 → 2"

9.21 DataFrame 반환

    return pd.DataFrame(values, index=labels, columns=time_bins)

NumPy 배열을 DataFrame으로 감싸 반환한다.

* 인덱스: "origin → destination"
* 열: bin 시작 시각의 누적 분
* 값: 해당 OD와 시간 bin에서 활성화된 요청 수

이 반환값은 save_heatmap()에 전달된다.

9.22 계산 복잡도

대략 다음 비용에 비례한다.

    O(OD 수 × 시간 bin 수 + 모든 요청이 걸치는 bin 수의 총합)

현재 기본 데이터는 75 × 158 행렬이고 실제 누적 횟수가 6,363회이므로 계산량은
크지 않다.


10. save_heatmap(): 132~183행
--------------------------------

정의:

    def save_heatmap(
        frequency: pd.DataFrame,
        request_count: int,
        time_bin: int,
        output_path: Path,
        dpi: int,
    ) -> None:

역할:

prepare_od_time_windows()가 만든 행렬을 Matplotlib 히트맵으로 그려 PNG로
저장한다. 파일을 만드는 함수이므로 별도의 반환값은 없다.

매개변수:

* frequency: OD × 시간 집계 DataFrame
* request_count: 제목에 표시할 전체 요청 수
* time_bin: x축 제목과 라벨 간격 계산에 사용할 분 간격
* output_path: PNG 저장 경로
* dpi: 출력 해상도

10.1 한글 글꼴

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]

Windows의 맑은 고딕을 우선 사용하고 DejaVu Sans를 대체 글꼴로 지정한다.
맑은 고딕이 없는 운영체제에서는 대체 글꼴의 한글 지원 여부에 따라 한글이
네모로 보일 수 있다.

10.2 음수 기호 설정

    plt.rcParams["axes.unicode_minus"] = False

일부 한글 글꼴에서 유니코드 마이너스 기호가 깨지는 것을 방지한다. 현재 집계
값에는 음수가 없지만 한글 Matplotlib 코드에서 흔히 사용하는 안전 설정이다.

10.3 NumPy 배열 변환

    values = frequency.to_numpy(dtype=int)

imshow()에 전달하기 쉽도록 DataFrame의 값을 2차원 정수 배열로 바꾼다. 행과
열 라벨은 빠지고 값만 남는다.

10.4 최댓값

    maximum = max(1, int(values.max()))

행렬 내 최대 활성 요청 수를 계산한다. 모든 값이 0이어도 최소 1을 사용해
색상표와 colorbar 단계가 비는 것을 방지한다. 기본 데이터의 실제 maximum은
10이다.

10.5 이산 색상표

    cmap = plt.colormaps["YlGnBu"].resampled(maximum + 1)

YlGnBu는 밝은 노랑에서 초록을 거쳐 파랑으로 진해지는 색상표다. 이를 최대값에
맞춰 이산 단계로 다시 샘플링한다.

10.6 0을 흰색으로 표시

    cmap.set_under("white")

최소 색상 경계를 0.5로 설정하므로 값 0은 under 범주가 된다. set_under로
under 색을 흰색으로 정해 활성 요청이 없는 셀을 명확히 구분한다.

10.7 정수별 색상 경계

    norm = BoundaryNorm(np.arange(0.5, maximum + 1.5), cmap.N)

maximum이 4라면 경계는 0.5, 1.5, 2.5, 3.5, 4.5다.

    0 → 0.5보다 작음 → 흰색
    1 → 0.5~1.5
    2 → 1.5~2.5
    3 → 2.5~3.5
    4 → 3.5~4.5

각 정수 값이 하나의 색상 단계에 정확히 대응한다.

10.8 그림 크기

    fig_width = max(16.0, frequency.shape[1] * 0.16)
    fig_height = max(12.0, frequency.shape[0] * 0.31 + 2.8)

열 수에 따라 너비, OD 행 수에 따라 높이를 늘린다.

기본 데이터에서는 다음과 같다.

    너비: 158 × 0.16 = 25.28인치
    높이: 75 × 0.31 + 2.8 = 26.05인치

300 DPI에서는 대략 7,584 × 7,815픽셀의 큰 이미지가 될 수 있다. 많은 OD
라벨을 읽을 수 있도록 크게 저장하는 설정이다.

10.9 figure와 axes 생성

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

fig는 그림 전체 객체이고 ax는 실제 히트맵과 축, 제목 등을 그리는 영역이다.

10.10 imshow

    image = ax.imshow(
        values,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

* values: 집계 행렬
* aspect="auto": 행과 열 수에 맞춰 셀 비율을 자동 조정
* cmap: YlGnBu 기반 이산 색상표
* norm: 정수별 색상 경계
* interpolation="nearest": 셀 사이를 부드럽게 보간하지 않고 단색 사각형 유지

10.11 x축 라벨 간격

    label_step = max(1, int(np.ceil(30 / time_bin)))

대략 30분마다 x축 시각 라벨을 표시한다.

    time_bin=5  → 6개 bin마다, 30분마다
    time_bin=10 → 3개 bin마다, 30분마다
    time_bin=15 → 2개 bin마다, 30분마다
    time_bin=20 → 2개 bin마다, 실제 40분마다
    time_bin=60 → 모든 bin, 실제 60분마다

time_bin이 항상 30의 약수는 아니므로 정확히 30분이 아니라 대략 30분 이상의
간격이 될 수 있다.

10.12 x축 tick 위치와 시각

positions는 라벨을 표시할 열 번호를 담고, minutes는 그 위치에 해당하는 누적
분을 가져온다.

누적 분은 다음 식으로 HH:MM 문자열로 되돌린다.

    시 = minute // 60
    분 = minute % 60

예를 들어 375는 06:15가 된다. :02d 형식은 한 자리 숫자 앞에 0을 붙이고,
라벨을 45도 회전해 서로 겹치는 것을 줄인다.

10.13 y축 라벨

모든 OD 행 위치에 frequency.index의 "origin → destination" 문자열을
표시한다. OD가 많으므로 글자 크기는 7.5로 비교적 작게 설정한다.

10.14 축 제목

기본값에서는 x축이 "시간대 (5분 간격)", y축이 "Origin → Destination"으로
표시된다.

10.15 그래프 제목

제목은 두 줄이다.

    시간대별 전체 요청 Time Window (OD별)
    총 요청 3,320건 · 색은 해당 시간대에 활성화된 time window 수

request_count에 :, 형식을 사용하므로 3320이 3,320으로 표시된다.

10.16 셀 경계선

imshow의 셀 중심은 0, 1, 2 같은 정수 좌표에 있고 셀 경계는 -0.5, 0.5,
1.5 등에 있다. 이 경계에 minor tick을 배치한 뒤 매우 얇은 흰색 grid를 그려
셀들을 구분한다. minor tick의 작은 눈금 표시는 숨긴다.

10.17 colorbar

히트맵 오른쪽에 색상 범례를 만든다. tick은 1부터 maximum까지의 모든 정수다.
값 0은 흰색이고 활성 요청 없음이라는 뜻이므로 colorbar tick에서 제외된다.
범례 제목은 "활성 Time Window 수"다.

10.18 레이아웃 정리

    fig.tight_layout()

제목, 축 라벨, tick 라벨이 이미지 밖으로 잘리지 않도록 여백을 자동 조정한다.

10.19 PNG 저장

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")

* output_path: 저장 위치
* dpi: 해상도
* bbox_inches="tight": 콘텐츠 바깥의 불필요한 여백 축소
* facecolor="white": 이미지 배경을 흰색으로 저장

10.20 figure 닫기

    plt.close(fig)

메모리에서 figure를 해제한다. 이 코드가 없으면 여러 그림을 반복 생성할 때
figure가 누적되어 메모리 사용량이 증가할 수 있다.


11. main(): 186~214행
------------------------

정의:

    def main() -> None:

역할:

명령행 인수 처리, 검증, CSV 읽기, 집계, 히트맵 저장을 순서대로 조정하는 실행
진입 함수다.

11.1 인수 읽기

    args = parse_args()

이후 args.input, args.output, args.time_bin, args.dpi로 각 설정에 접근한다.

11.2 time_bin 검증

    if args.time_bin <= 0 or 60 % args.time_bin != 0:
        raise ValueError("--time-bin은 60의 약수인 양의 정수여야 합니다.")

0이나 음수, 또는 7, 11, 14처럼 60의 약수가 아닌 값을 거부한다. 각 시간의
경계와 bin이 규칙적으로 맞아떨어지도록 하는 제한이다.

11.3 DPI 검증

    if args.dpi <= 0:
        raise ValueError("--dpi는 양의 정수여야 합니다.")

argparse가 정수 변환은 이미 수행하므로 여기서는 양수 여부를 확인한다.

11.4 입력 경로 정규화

    input_path = args.input.expanduser().resolve()

expanduser()는 ~를 사용자 홈 폴더로 확장하고, resolve()는 상대 경로와 . 및
..을 정리한 절대 경로를 만든다.

사용자가 전달한 상대 --input 경로는 현재 터미널 작업 폴더 기준이다. 기본
입력 경로는 __file__ 기준으로 구성되므로 현재 작업 폴더와 무관하다.

11.5 입력 파일 확인

    if not input_path.is_file():

경로가 존재하지 않거나 폴더라면 FileNotFoundError를 발생시킨다.

11.6 출력 경로 결정

--output을 지정했으면 그 경로를 확장하고 절대 경로로 변환한다. 지정하지
않았으면 입력 파일과 같은 폴더에서 입력 stem 뒤에
_all_time_windows_od.png를 붙인다.

    rolling_horizon_request_status.csv
    → rolling_horizon_request_status_all_time_windows_od.png

11.7 출력 확장자 검증

    if output_path.suffix.casefold() != ".png":

PNG가 아닌 출력 경로를 거부한다. casefold()를 사용하므로 .png, .PNG, .Png는
모두 허용된다. 확장자가 없거나 .jpg, .pdf이면 오류다.

11.8 출력 폴더 생성

    output_path.parent.mkdir(parents=True, exist_ok=True)

출력 폴더가 없으면 중간 상위 폴더까지 생성한다. 이미 있어도 오류를 내지 않는다.
이 작업은 CSV 내용 검증보다 먼저 수행되므로 이후 오류가 발생해도 새로 만든 빈
폴더는 남을 수 있다.

11.9 CSV 읽기

    data = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig")

dtype=str은 열을 문자열로 읽게 하여 "001" 같은 노드 번호가 1로 바뀌는 문제를
줄인다. encoding="utf-8-sig"는 일반 UTF-8과 BOM이 포함된 UTF-8 CSV를
읽으며, 첫 열 이름 앞에 BOM 문자가 붙는 문제를 방지한다.

11.10 필수 열 검사

    missing = REQUIRED_COLUMNS.difference(data.columns)

필수 열 중 실제 CSV에 없는 열의 집합을 만든다. 누락 열이 있으면 정렬된 목록을
KeyError 메시지로 보여준다. 열 이름은 대소문자와 공백까지 정확히 일치해야 한다.

11.11 집계 행렬 생성

    frequency = prepare_od_time_windows(data, args.time_bin)

기본 데이터에서는 shape가 (75, 158)인 DataFrame이 생성된다.

11.12 히트맵 저장

    save_heatmap(frequency, len(data), args.time_bin, output_path, args.dpi)

전체 요청 수 len(data)는 그림 제목에 사용된다. 집계 행렬, 시간 간격, 저장
경로, DPI를 save_heatmap()에 넘겨 PNG를 만든다.

11.13 실행 결과 출력

기본 데이터에서는 다음 형식으로 출력한다.

    전체 요청: 3,320건
    OD 조합: 75개
    PNG 저장 완료: C:\...\rolling_horizon_request_status_all_time_windows_od.png


12. 스크립트 진입 조건: 217~218행
------------------------------------

    if __name__ == "__main__":
        main()

파일을 다음처럼 직접 실행하면 __name__이 "__main__"이 되어 main()을
호출한다.

    python ".\코드\PDPTW_main\data정리\request_od.py"

다른 Python 파일에서 import하면 __name__은 모듈 이름이 되므로 main()은
자동 실행되지 않는다.

    import request_od

따라서 import 시 PNG가 갑자기 생성되지 않으며, 다음 함수들을 개별적으로
재사용할 수 있다.

    request_od.hhmm_to_minutes(...)
    request_od.prepare_od_time_windows(...)
    request_od.save_heatmap(...)


13. 기본 및 사용자 지정 실행 방법
-----------------------------------

13.1 기본 실행

저장소 루트에서 다음과 같이 실행한다.

    python ".\코드\PDPTW_main\data정리\request_od.py"

기본 설정:

* 입력: DEFAULT_INPUT의 rolling_horizon_request_status.csv
* 시간 간격: 5분
* DPI: 300
* 출력: 입력 CSV 폴더의 rolling_horizon_request_status_all_time_windows_od.png

13.2 사용자 지정 실행

PowerShell 예시:

    python ".\코드\PDPTW_main\data정리\request_od.py" `
      --input ".\자료\요청.csv" `
      --output ".\결과\request_od_heatmap.png" `
      --time-bin 10 `
      --dpi 200


14. 히트맵 해석 방법
--------------------

예를 들어 다음 셀이 있다고 가정한다.

    행: 4 → 9
    열: 06:20
    값: 3

의미는 다음과 같다.

    origin=4, destination=9인 요청 중 time window가
    06:20~06:25 구간과 겹치는 요청이 3개다.

이 값은 다음을 의미하지 않는다.

* 실제 탑승 요청 수
* 해당 시각에 운행 중인 차량 수
* 실제 pickup 횟수
* overdue 요청 수
* 요청이 겹친 시간의 총 길이
* 배차 완료 건수

오직 입력 CSV에 있는 요청 time window의 동시 활성 개수다.


15. 시간 경계 처리 예시
-----------------------

모든 예시는 5분 bin을 가정한다.

15.1 정확히 한 bin에 들어가는 요청

    요청: 06:15~06:20

    06:15~06:20: +1
    06:20~06:25: +0

종료 시각은 제외하므로 다음 bin에 들어가지 않는다.

15.2 두 bin에 걸치는 요청

    요청: 06:17~06:22

    06:15~06:20: +1
    06:20~06:25: +1

15.3 짧지만 경계를 넘는 요청

    요청: 06:19~06:21

실제 길이는 2분이지만 두 bin과 겹친다.

    06:15~06:20: +1
    06:20~06:25: +1

15.4 bin 경계와 정렬된 요청

    요청: 06:20~06:25

    06:15~06:20: +0
    06:20~06:25: +1
    06:25~06:30: +0

15.5 길이가 0인 요청

    요청: 06:20~06:20

특수 처리로 06:20~06:25 bin에 +1이 된다.


16. 코드상 주의점과 잠재적인 예외
------------------------------------

16.1 빈 CSV

필수 열은 있지만 데이터 행이 0개이면 start.min()과 end.max()가 NaN이 되어
정수 변환 단계에서 오류가 발생할 수 있다. 현재 코드는 적어도 한 개의 유효
요청이 있다는 것을 전제로 한다.

16.2 prepare_od_time_windows()의 time_bin 검증

time_bin이 양수이고 60의 약수인지 확인하는 코드는 main()에만 있다. 외부
코드가 prepare_od_time_windows(data, 0)처럼 직접 호출하면 0으로 나누기 등의
오류가 날 수 있다. 정상 CLI 실행에서는 main()이 먼저 검증한다.

16.3 자정 통과 요청

23:55 → 00:10은 종료 분이 시작 분보다 작아져 오류가 된다. 날짜와 다중 일자
시간창을 지원하지 않는다.

16.4 날짜와 초 단위 정보 미사용

다른 날짜 열이나 초 단위 열이 있어도 time_window_start_hhmm과
time_window_end_hhmm만 사용한다. 서로 다른 날짜의 요청을 하나의 CSV에 넣으면
같은 하루의 시간축에 합쳐진다.

16.5 중복 제거 없음

selected_request_id나 batch_id를 이용한 중복 제거가 없다. 같은 요청이 두
행에 있으면 두 번 집계된다.

16.6 상태 필터링 없음

final_status는 사용하지 않는다. Complete, Overdue 등 모든 행이 포함된다.

16.7 방향성 OD

1 → 2와 2 → 1은 별도의 행이다.

16.8 겹친 길이를 반영하지 않음

어떤 요청이 한 bin과 1분 겹치든 5분 겹치든 해당 셀에는 동일하게 1을 더한다.
이 히트맵은 활성 시간의 총량이 아니라 활성 time window 개수를 보여준다.

16.9 노드 값 정규화 없음

"1", "01", "1.0"은 서로 다른 문자열이므로 별도 OD가 될 수 있다. 숫자
변환은 정렬 키에만 적용된다.

16.10 전역 시간 범위 유지

가장 이른 시작 bin부터 가장 늦은 종료 경계까지 모든 열을 만든다. 중간에 어느
OD에도 요청이 없는 시간이 있어도 열을 제거하지 않고 흰색으로 표시한다.

16.11 정확히 bin 경계에 있는 0분 요청의 잠재적 경계 문제

데이터 전체의 가장 늦은 종료 시각이 정확히 19:20이고 동시에 19:20 → 19:20
요청이 있다고 가정한다. 전체 time_bins를 만들 때 last_edge=19:20은 range의
끝값이므로 19:20 열이 생성되지 않을 수 있다. 하지만 개별 요청 처리에서는
0분 요청을 한 bin에 넣기 위해 19:20 bin을 요구한다. 이 경우
bin_index[minute]에서 KeyError가 날 가능성이 있다.

즉, 시작과 종료가 같은 요청을 한 bin에 표시하는 개별 로직과 전체 시간축의
마지막 경계 계산이 완전히 일치하지 않는 드문 corner case가 있다. 일반적인
양의 길이 time window에서는 문제가 없다.


17. 함수별 책임 요약
---------------------

parse_args()
    명령행 옵션을 해석한다.
    입력: 터미널 인수
    출력: argparse.Namespace

hhmm_to_minutes()
    HH:MM을 자정 이후 누적 분으로 변환한다.
    입력: 시간 Series, 오류 표시용 열 이름
    출력: 분 단위 Series

sortable_node()
    숫자 및 문자열 노드를 자연스럽게 정렬할 키를 만든다.
    입력: 노드 값
    출력: 정렬용 tuple

prepare_od_time_windows()
    요청들을 OD × 시간 활성 개수 행렬로 집계한다.
    입력: CSV DataFrame, 시간 bin 크기
    출력: 집계 DataFrame

save_heatmap()
    집계 행렬을 히트맵 PNG로 저장한다.
    입력: 집계 DataFrame, 요청 수, bin 크기, 출력 경로, DPI
    출력: 없음. 파일 시스템에 PNG 생성

main()
    검증, CSV 읽기, 집계, 이미지 저장의 전체 실행 순서를 조정한다.
    입력: 없음. parse_args()가 명령행 인수를 읽음
    출력: 없음. 요약 메시지 출력 및 PNG 생성


18. 전체 평가
-------------

이 파일은 역할이 비교적 명확히 분리되어 있다.

* 시간 문자열 변환: hhmm_to_minutes()
* 노드 정렬: sortable_node()
* 핵심 집계: prepare_od_time_windows()
* 시각화 및 저장: save_heatmap()
* 실행 제어와 파일 입출력: main()

핵심 해석은 "한 요청의 [start, end) time window와 실제로 겹치는 모든 시간
bin에 1씩 더한다"는 것이다. 따라서 히트맵의 진한 셀은 해당 OD에서 그 시간대에
허용 시간창이 동시에 많이 열려 있음을 뜻한다.

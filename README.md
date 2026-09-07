1. 핵심코드
   - 코드\PDPTW_main\time_limit\PDPTW_NEW_Remove_Depot_test.py -> 초기 Depot과 관계없이 최종 마지막 도착지가 Depot이 되는 방식
   - 코드\PDPTW_main\time_limit\PDPTW_NEW_TEST.py -> 초기 Depot으로 복귀
2. 관련 출력 결과물
   - 자료\결과\Ortools
   - Rolling_Horizon_N: Rolling Horizon의 구간 설정을 의미합니다. ex) Rolling_Horizon_30: [0,30]
   - dN: 각 수요의 수에 따른 결과(현재는 5000명을 중심으로 결과를 확인하였습니다.)
   - Penalty_Per_Vehicles
   - Time_Solver or (Original, Remove_Depot): Time_Solver 폴더의 경우는 Max_wait()를 확장한 경우고 그 외에는 Max_wait()를 전혀 건들지 않은 경우입니다.
     
3. 참고한 논문 자료
   - 자료\기초자료\참고자료에 있습니다!
  
그 외에 시각화 및 data 정리를 할 때 활용한 코드는 모두 PDPTW\코드\PDPTW_main\data정리에 위치해 있습니다. 

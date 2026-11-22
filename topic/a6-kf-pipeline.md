> [!info]
>  KFP `task` (`PipelineTask`) 객체란?
> 
> `@dsl.pipeline` 내부에서 컴포넌트(`@dsl.component`)를 호출할 때 반환되는 **파이프라인 단계(Step) 객체**입니다. 함수가 즉시 실행되는 것이 아니라, 쿠버네티스 파드(Pod)로 실행하기 위한 명세서 역할을 합니다.
>
> ### 💡 사용 예시
> ```python
> @dsl.pipeline(name="llm-finetuning-pipeline")
> def my_pipeline(raw_path: str):
>     # 1. 컴포넌트 호출 시 결과값이 아닌 PipelineTask 객체 반환
>     prep_task = preprocess_data(raw_path=raw_path)
>     
>     # 2. cpu() 헬퍼 함수로 자원 및 기본 캐시(cache=True) 설정
>     cpu(prep_task)
>     
>     # 3. 알림·관문·배포 단계는 매번 실행되도록 cache=False로 덮어쓰기
>     alert_task = send_slack_alert(message="학습 완료!")
>     cpu(alert_task, memory="1Gi", cache=False)
> ```
>
> ### 🛠️ `task` 객체로 제어하는 항목
> * **리소스 할당**: `set_cpu_request()`, `set_memory_limit()` 등으로 파드 자원 설정
> * **캐싱 제어**: `set_caching_options()`로 이전 실행 결과 재사용 여부 결정
> * **시크릿 주입**: `kubernetes.use_secret_as_env()`로 API 키 및 환경변수 전달
> * **실행 순서**: `task_b.after(task_a)` 형태로 작업 간 의존성 및 순서 지정

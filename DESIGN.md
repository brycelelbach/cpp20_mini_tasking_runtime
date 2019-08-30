Idea: User-space tasking system with future/promises used to implement parallel
algorithms.

```
jthread & stop_token - worker threads
semaphores - task queue (producer/consumer problem)
atomic wait/notify - obstruction-free unique future/promise
floating point atomics - performance counters (average time per task)
latch - startup, probably
barrier - ? (use in the "application" running on the system)
atomic_ref - concurrent accumulation into user data (possibly scan?)
```

https://en.wikipedia.org/wiki/Semaphore_(programming)
https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem

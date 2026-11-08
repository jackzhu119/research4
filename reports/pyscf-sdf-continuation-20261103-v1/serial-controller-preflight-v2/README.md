# Final controller preflight

This v2 receipt tests the frozen controller source
`d0a67096215102a6e7b46ce9510e1eee6e3e81517de5f3e09e5f86f44c13bd6e`.
All 16 pure engineering tests actually passed. No QM process was launched.
The earlier `serial-controller-preflight/` receipt remains unchanged and records
the preceding 15-test source version.

Usage and archive/recovery contracts are described in
`../serial-controller-preflight/README.md`. The only later runtime addition is
a finite outer supervisor-wall watchdog: if the frozen V4 supervisor has not
exited by its granted wall limit plus two seconds, the controller requests TERM,
waits at most three seconds, then requests KILL and waits at most three seconds.
V4's actual child-limits contract arms parent-death SIGKILL. Any such event is a
failure/partial outcome; no healthy exit, final CPU cost or completed stage is
invented, and no further phase starts. The watchdog test mocks Popen, clocks and
signals completely and runs no process.

The controller performs no Git operation. Root still owns actual gate release,
scientific phase execution, every report/checkpoint backup, and continuation
after any failed or forensic-reviewed phase.

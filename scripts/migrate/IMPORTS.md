# Import log

Append-only record of every history import and re-sync performed with
`scripts/migrate/extract-sdk-python.sh`. The merge SHA is the `--allow-unrelated-histories`
merge commit on `main`.

| Date (UTC) | Plugin | Source repo @ SHA | git-filter-repo | Merge SHA | Notes |
|---|---|---|---|---|---|
| 2026-09-02 | openai_agents | temporalio/sdk-python @ d0e075c1e25b3371f0e0f5116d9c6626892e3eab | 2.47.0 | 685eb28a2af86eff456793647fbc04afeacddb09 | initial import: 101 commits, 18 identities, root 53d9ace6 upstream |
| 2026-09-15 | openai_agents | temporalio/sdk-python @ b5a2bef1522da1825d931821a8b99e6492e2eac2 | 2.47.0 | 4e572c68f3a4651fda57cdf8764f8256ced548f7 | re-sync: 104 commits, 18 identities; retained the local offline OpenAI mock while accepting upstream timeout removals |
| 2026-10-01 | openai_agents | temporalio/sdk-python @ 65fecc416588151b65e62495d76015778124dd8d | 2.47.0 | a026834524e925af20629deed638244becd5bc80 | final re-sync requested after cutover: 105 commits, 18 identities; source is the parent of removal commit 6e66be6ea30537e4536a54d1c9a92ff0052bb42d; imported untraced tracing-test waits and ported their helper support while preserving the new package/test paths, MCP v2 adapter, offline mocks and local fixes |
| 2026-10-05 | temporal-spring-ai | temporalio/sdk-java @ be01e60acc1e2ccfb20e783a9770bad745ed85c1 | 2.47.0 | 639895dfd571f75d6b4c56c38b77b98d55911450 | initial Java import: 12 rewritten commits, 1 identity; 13 upstream path commits, with the directory-only rename pruned after both paths map to the same destination; upstream build and README archived in `_upstream/`; imported issue and PR references qualified as `temporalio/sdk-java#NNN` before the initial import merged |

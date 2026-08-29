# Post-oracle fork receipt v1

This language-neutral bundle records an actual `activegraph-bridge` fork after
a committed, recorded oracle effect. The parent executes one deterministic
offline fixture oracle call. Verification and the child fork serve that result
from the log; the fork performs zero inherited external calls. A changed tool
result creates the divergent child tail.

`receipt.json` binds the source prefix, child log, inherited effect identity,
source and target fingerprints, and signed target-environment claims. The HMAC
key is deliberately published because this is a conformance trust root, not a
production credential or claim about a real provider environment.

Run from the repository root:

```sh
./evidence/post-oracle-fork-v1/verify.sh
```

Expected result:

```text
POST-ORACLE FORK RECEIPT PASS — committed oracle served from record; 0 inherited external calls
```

The fixture makes no model-quality or provider-authenticity claim.

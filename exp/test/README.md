Multi-stage experiments for testing (only manual testing so far).

To generate data, run, for example

```
for i in `seq 1 1000`; do echo $i | md5sum; done >train.txt
```


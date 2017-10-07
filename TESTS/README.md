*Note*:
In order to set a breakpoint in the code, copy and paste the following statement:
```Python
import code
code.interact(local=locals())
```
This works in Python2.7 only when the code is run in serial. That is, you need to set `self.cores` to 1.


import os
import itertools

lrs = [1e-4, 2e-4, 5e-4, 1e-5, 2e-5, 5e-5, 1e-6]
wds = [1e-5, 1e-4, 1e-3, 1e-2]
wms = [25, 50, 100, 150]
nes = [30, 40, 50]


f = open('run.sh', 'w')
for args in itertools.product(lrs, wds, wms, nes):
  f.write(f'indication_setup {args[0]} {args[1]} {args[2]} {args[3]} 16\n')
f.close()
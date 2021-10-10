#### Linpack output:

Sample data file lininput_xeon64.

Current date/time: Sun Oct 10 22:18:55 2021

CPU frequency:    4.499 GHz
Number of CPUs: 1
Number of cores: 6
Number of threads: 6

Parameters are set to:

Number of tests: 5
Number of equations to solve (problem size) : 1000  2000  5000  10000 20000
Leading dimension of array                  : 1000  2000  5008  10000 20000
Number of trials to run                     : 4     2     2     2     1    
Data alignment value (in Kbytes)            : 4     4     4     4     4    

Maximum memory requested that can be used=3200404096, at the size=20000

=================== Timing linear equation system solver ===================

Size   LDA    Align. Time(s)    GFlops   Residual     Residual(norm) Check
1000   1000   4      0.003      205.4947 9.394430e-13 3.203742e-02   pass
1000   1000   4      0.004      189.0014 9.394430e-13 3.203742e-02   pass
1000   1000   4      0.003      207.4709 9.394430e-13 3.203742e-02   pass
1000   1000   4      0.003      219.0231 9.394430e-13 3.203742e-02   pass
2000   2000   4      0.048      111.3705 3.842024e-12 3.342090e-02   pass
2000   2000   4      0.046      115.9297 3.842024e-12 3.342090e-02   pass
5000   5008   4      0.336      248.4300 2.313949e-11 3.226615e-02   pass
5000   5008   4      0.335      249.0125 2.313949e-11 3.226615e-02   pass
10000  10000  4      2.352      283.5113 9.955517e-11 3.510416e-02   pass
10000  10000  4      2.395      278.4902 9.955517e-11 3.510416e-02   pass
20000  20000  4      18.362     290.4959 3.520981e-10 3.116839e-02   pass

Performance Summary (GFlops)

Size   LDA    Align.  Average  Maximal
1000   1000   4       205.2475 219.0231
2000   2000   4       113.6501 115.9297
5000   5008   4       248.7213 249.0125
10000  10000  4       281.0007 283.5113
20000  20000  4       290.4959 290.4959

Residual checks PASSED

End of tests
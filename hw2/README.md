1. Реализовать классическое перемножение матриц и умножение матрицы на вектор на C/C++: Реализация в файле __matmul.cpp__

2. Разбейте на модули, со статической линковкой скомпилируйте текст, подготовьте Makefile, проверьте флаги -g, -O3: файлы __matmul.cpp__ __main.cpp__ __Makefile__ в директории. -O3 или -g передаются через флаг __EXTFLAGS__

#### command: make

        Matrix mult: ijk run time is: 0.75526

        Matrix mult: jik run time is: 0.779777

        Matrix mult: kij run time is: 0.401694

        Matrix to vector mult: run time is: 0.00355

#### command: make -EXTFLAGS='-O3' 

        Matrix mult: ijk run time is: 0.262969

        Matrix mult: jik run time is: 0.268058

        Matrix mult: kij run time is: 0.046232

        Matrix to vector mult: run time is: 0.00123


3. Измерьте времена исполнения для размеров N = 500, 512, 1000, 1024, 2000, 2048 
Все команды выполнены с флагом __O3__
###### N = 500:
        Matrix mult: ijk run time is: 0.149832
        
        Matrix mult: jik run time is: 0.130572
        
        Matrix mult: kij run time is: 0.045606
        
        Matrix to vector mult: run time is: 0.000624
        
###### N = 512:
        Matrix mult: ijk run time is: 0.278448
        
        Matrix mult: jik run time is: 0.267043
        
        Matrix mult: kij run time is: 0.045168
        
        Matrix to vector mult: run time is: 0.000275

###### N = 1000:
        Matrix mult: ijk run time is: 1.3334
        
        Matrix mult: jik run time is: 1.2496
        
        Matrix mult: kij run time is: 0.344876
        
        Matrix to vector mult: run time is: 0.001065
        
###### N = 1024:
        Matrix mult: ijk run time is: 2.44418
        
        Matrix mult: jik run time is: 2.28913
        
        Matrix mult: kij run time is: 0.388196
        
        Matrix to vector mult: run time is: 0.001221
        
###### N = 2000:
        Matrix mult: ijk run time is: 28.6923
        
        Matrix mult: jik run time is: 27.0391
        
        Matrix mult: kij run time is: 4.91789
        
        Matrix to vector mult: run time is: 0.004674
        
###### N = 2048:
        Matrix mult: ijk run time is: 45.9769
        
        Matrix mult: jik run time is: 36.4702
        
        Matrix mult: kij run time is: 6.20823
        
        Matrix to vector mult: run time is: 0.004632

4. И базовые скрипты баш: Скрипты в папке __bash_scripts__
5. Бонус за линпак: результат в файле __linpack_out.png__

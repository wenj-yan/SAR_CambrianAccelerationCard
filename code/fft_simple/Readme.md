- Basic FFT test code:

```
-fft_kt_samole.cc    	based on MLU370
-fft_test.cu			based on cuda(4060Laptop)
-fft_test.m             based on matlab
```

- test result:

|    code_name     |     device      | time_All(ms) | time_H2D | time_D2H | time_compute |
| :--------------: | :-------------: | :----------: | :------: | :------: | :----------: |
|    fft_test.m    | CPU(I9-14900HX) |     665      |    -     |    -     |      -       |
|   fft_test.cu    | GPU(4060Laptop) |     180      |    67    |   104    |      9       |
| fft_kt_samole.cc |   NPU(MLU370)   |      70      |    5     |    63    |      3       |

- test method:

```
-fft_kt_samole.cc    
	  # 参照用户手册中“部署CNNL”一节中内容配置CNtookit，获得neuware文件夹
  cd cnnl_fft_sample
  export NEUWARE_HOME=/usr/local/neuware # 设置你的neuware环境，请务必正确设置
  cd .. #退回上一级目录
  source env.sh # 设置环境变量，指定库路径
  cd fft_kt_sample
  make clean
  make # 生成可执行文件conv_sample
  
-fft_test.cu	
	nvcc -o fft_test fft_test.cu -lcufft
	
-fft_test.m  
	Run directly with matlab
```


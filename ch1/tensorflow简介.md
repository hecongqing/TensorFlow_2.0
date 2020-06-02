# TensorFlow 2.0简介



# TensorFlow 2.0安装

## Ubuntu下安装

### 1. Anaconda 安装
bash Anaconda2-4.4.0-Linux-x86_64.sh
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```

### 2、nvidia驱动安装
方法一（容易安装不成功）：第一步：卸载可能存在的旧版本 nvidia 驱动

``` she
sudo apt-get remove nvidia-*
sudo apt-get autoremove
```



第二步：输入CTRL+ALT+F1进入文本模式

第三步：临时关闭显示服务

```
sudo service lightdm stop
```

第四步：重新安装Nvidia驱动

```
sudo ./NVIDIA-Linux-x86_64-415.13.run -no-x-check -no-nouveau-check -no-opengl-files
-no-x-check安装驱动时关闭x服务;
-no-nouveau-check 安装驱动时禁用Nouveau
-no-opengl-files 安装时只装驱动文件，不安装Opengl
```

第五步：启动显示服务（自动跳转到桌面）

```
sudo service lightdm restart
```

第六步：查看Nvidia驱动是否安装成功

```
nvidia-smi
```

方法二：
系统设置->软件更新->附加驱动->选择nvidia最新驱动(361)->应用更改


方法三：

第一步：卸载可能存在的旧版本 nvidia 驱动

```
sudo apt-get remove nvidia-*
sudo apt-get autoremove
```



第二步：输入CTRL+ALT+F1进入文本模式

第三步：临时关闭显示服务

```
sudo service lightdm stop
```

第四步、禁用nouveau驱动
Ubuntu系统集成的显卡驱动程序2是nouveau，我们需要先将nouveau从linux内核卸载掉才能安装NVIDIA官方驱动。
将nouveau添加到黑名单blacklist.conf中，(关于blacklist参见 《禁用Linux内核驱动》),linux启动时，就不会加载nouveau.
因为nouveau驱动的影响，ubuntu安装后无法登入桌面，所以在ubuntu系统启动显示登录界面后，需要按ctrl+alt+F1进入tty文本模式进入下面的操作

由于blacklist.conf文件的属性不允许修改。所以需要先修改文件属性。
查看属性

```
ll /etc/modprobe.d/blacklist.conf
```



修改属性

```
sudo chmod 666 /etc/modprobe.d/blacklist.conf
```




用vi编辑器打开

```
sudo vi /etc/modprobe.d/blacklist.conf
```



在文件末尾添加如下几行：

```
blacklist vga16fb
blacklist nouveau
blacklist rivafb
blacklist rivatv
blacklist nvidiafb
```



修改并保存文件后，记得把文件属性复原：

```
sudo chmod 644 /etc/modprobe.d/blacklist.conf
```



再更新一下内核

```
sudo update-initramfs -u
```



关于update-initramfs命令的用途，参见 《initramfs 简介，一个新的 initial RAM disks 模型》
修改后需要重启系统。
重启系统确认nouveau是否已经被屏蔽掉，使用lsmod命令查看：

```
lsmod | grep nouveau
```



lsmod命令用于显示已经加载到内核中的模块的状态信息，参见《lsmod命令》



第五步：添加ppa库，通过ppa安装显卡驱动，注意不要从NVIDIA官网下载显卡驱动，直接通过ppa安装即可：

```
sudo add-apt-repository ppa:graphics-drivers/ppa

sudo apt-get update

ubuntu-drivers devices

sudo apt-get install nvidia-430
```



注意： 如果 sudo apt-get update 很慢，

```
sudo vim /etc/apt/sources.list
```



用这里面的源进行替换: https://mirrors.tuna.tsinghua.edu.cn/help/ubuntu/

第六步：查看Nvidia驱动是否安装成功

```
nvidia-smi
```






### 3、CUDA安装

```
sudo sh cuda_10.0.130_410.48_linux.run
```




环境变量加入：

```
export CUDA_HOME=/usr/local/cuda-10.0

export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH

export PATH=/usr/local/cuda-10.0/bin:$PATH
```




### 4、cuDNN安装

```
tar -xzf cudnn-8.0-linux-x64-v5.1.tgz

cd cuda

sudo cp lib64/* /usr/local/cuda/lib64/

sudo cp include/* /usr/local/cuda/include/
```


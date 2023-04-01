# WSL迁移、更换wsl的默认位置

> 下列命令均在`powershell`中执行：

## 1. 关闭wsl

```powershell
wsl --shutdown
```

## 2. 导出WSL分发版

我的分发版本是：ubuntu；迁移的地址为 F:\WSL\images\my_ubuntu.tar

```powershell
wsl --export Ubuntu F:\WSL\images\my_ubuntu.tar
```

## 3. 卸载当前分发版

```powershell
wsl --unregister Ubuntu
```

## 4. 导入tar文件

```powershell
 wsl --import Ubuntu F:\WSL\my_ubuntu F:\WSL\images\my_ubuntu.tar
```

## 5. 设置默认用户

your_name是你安装wsl时使用的用户名

```powershell
ubuntu config --default-user your_username
```


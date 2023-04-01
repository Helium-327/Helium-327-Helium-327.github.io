# WSL配置`zsh`

## zsh

### **安装**：

```bash
sudo apt-get install zsh
```

### **查看可以用的**`shell`

```bash
cat /etc/shells
```

### **更改你的默认** `Shell`

```bash
chsh -s /bin/zsh
```

## oh-my-zsh

### **安装**：

```bash
# 用curl
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

或者

```bash
# 用wget
sh -c "$(wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"
```

### **修改主题**

内置主题已经放在 `～/.oh-my-zsh/themes` 目录下

想要修改使用 vim 编辑 `.zshrc`，键入以下内容并保存：

```bash
ZSH_THEME=" "
```

### 安装插件

#### zsh-autosuggestions

* 把插件下载到本地的 `~/.oh-my-zsh/custom/plugins` 目录：

```bash
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
```

* 在 `.zshrc` 中，把 `zsh-autosuggestions` 加入插件列表：

```bash
plugins=(
    # other plugins...
    zsh-autosuggestions  # 插件之间使用空格隔开
)
```

* 开启新的 Shell 或执行 `souce ~/.zshrc`，就可以开始体验插件。

#### zsh-syntax-highlighting

* 把插件下载到本地的 `~/.oh-my-zsh/custom/plugins` 目录:

```bash
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting 
```

* 在 `.zshrc` 中，把 `zsh-syntax-highlighting` 加入插件列表：

```bash
plugins=(
    # other plugins...
    zsh-autosuggestions
    zsh-syntax-highlighting
)
```

* 开启新的 Shell 或执行 `souce ~/.zshrc`，就可以开始体验插件了。

#### z

* 由于 oh-my-zsh 内置了 z 插件，所以只需要在 `.zshrc` 中，把 z 加入插件列表：

```bash
plugins=(
     # other plugins...
     zsh-autosuggestions
     zsh-syntax-highlighting
     z
)
```

* 开启新的 Shell 或执行 `souce ~/.zshrc`，就可以开始体验插件了。

### 设置 alias

类似于`cd ~/projects/alicode/blog`命令这种：

* 在 `.zshrc` 中键入：

```bash
alias cdblog="cd ~/projects/alicode/blog" 
```

* 开启新的 Shell 或 `souce ~/.zshrc`，以使配置生效。生效后就可以使用 `cdblog` 进行跳转了
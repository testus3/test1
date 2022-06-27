### 3. Pick interval with condition and drop NA values
Tasks:

  3.1   Pick values of y (mpg) so that its acceleration `acc>25`<br>
  3.2  List items and drop the ones who don't have a value or are empty

Example: Pick all <img src="https://latex.codecogs.com/svg.image?{\large\color{Blue}\pmb{a_n|b_n>2}" align="center">


$$
a_n \quad b_n\\
\begin{bmatrix}
6 & 2\\
8 & 4\\
4 & 8
\end{bmatrix}
$$


Then <img src="https://latex.codecogs.com/svg.image?{\large\color{Blue}\pmb{a_n=\{8,4\}}" align="center">, in programming to calculate the mean of such <img src="https://latex.codecogs.com/svg.image?{\large\color{Blue}\pmb{a_n}" align="center"> with that condition, it'll be `print(np.mean(an[bn>2]))` 




```python
val = np.array([[6,2],[8,4],[4,8]])
xn = val[:,0]
print(xn)
yn = val[:,1]
I = (yn>2)
print(I)
print(np.mean(xn*I)/np.mean(I))
print(np.mean(xn*I))
print(np.mean(xn[yn>2]))
```

    [6 8 4]
    [False  True  True]
    6.0
    4.0
    6.0
    


```python
#3.1 Pick values of y (mpg) so that its acceleration acc>25
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
acc = np.array(fd['acceleration'])
plt.plot(y,acc,'o')
print('Mean of mpg | acc>25: %.2f' %np.mean(y[acc>15]))
```

    Mean of mpg | acc>25: 25.85
    





```python
!jupyter nbconvert --execute --to markdown Lec02.ipynb
```

    [NbConvertApp] Converting notebook Lec02.ipynb to markdown
    [NbConvertApp] Support files will be in Lec02_files\
    [NbConvertApp] Making directory Lec02_files
    [NbConvertApp] Making directory Lec02_files
    [NbConvertApp] Writing 6520 bytes to Lec02.md
    

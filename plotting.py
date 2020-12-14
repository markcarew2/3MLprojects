xList = np.arange(-2, 2,.666 * .1)
xList = xList[0:60]
yList = np.arange(-2,4,.1)
X, Y = np.meshgrid(xList,yList)

Z = X * Bs[0] + Y * Bs[1] + Bs[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.plot(dfNormalized[0],dfNormalized[1], labels,  'ro')
ax.plot_surface(X,Y,Z)
plt.show(block=True)
#plt.pause(.5)
#plt.close()



#PROBLEM 3 PLOTTING
df0 = df[df["label"] == 0]
df1 = df[df["label"] == 1]

plt.plot(df1["A"],df1["B"], "ro")
plt.plot(df0["A"],df0["B"],"bo")

plt.show(block = False)
plt.pause(1)
plt.close()
import matplotlib.pyplot as plt

number = input("2 for len score, 3 for orientation score, 4 for total score")

if number == '1' :
    fp = open("Episode-constrain-score.txt", "r")
    print("now showing Episode-constrain-score")
elif number == '2' :
    fp = open("Episode-len-score.txt", "r")
    print("now showing Episode-len-score")
elif number == '3' :
    fp = open("Episode-orientation-score.txt", "r")
    print("now showing Episode-orientation-score")
else :
    fp = open("Episode-score.txt", "r")
    print("now showing Episode-score")
    #fp = open("Episode-orientation-score.txt", "r")
    #print("now showing Episode-orientation-score")
    #fp = open("Episode-len-score.txt", "r")
    #print("now showing Episode-len-score")
    #fp = open("Episode-constrain-score.txt", "r")
    #print("now showing Episode-constrain-score")

lines = fp.read().splitlines()
scores = list(map(float, lines))
episode = list(range(1, 1+len(scores)))
#print(scores)
#print(episode)
plt.figure()
plt.plot(episode, scores)
plt.xlabel("Episode",fontsize=13,fontweight='bold')
plt.ylabel("Score",fontsize=13,fontweight='bold')
#plt.show()

if number == '1' :
    plt.savefig('trend_constrain.png')
elif number == '2' :
    plt.savefig('trend_len.png')
elif number == '3' :
    plt.savefig('trend_orientation.png')
else :
    plt.savefig('trend_total.png')
    
fp.close()
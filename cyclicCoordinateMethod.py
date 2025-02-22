from sympy import *
import time

def print_box(content):
    content_str = f"║ {content} ║"
    width = len(content_str)
    print("▀" * 50) 
    print("╔" + "═" * (width - 2) + "╗")
    print(content_str)
    print("╚" + "═" * (width - 2) + "╝")


def f():
    X1,X2 = symbols("X1 X2")
    F = (X1-2)**4 + (X1 - 2*X2)**2
    return ((X1,X2),F)

def fSub(subX1,subX2):
    (X1,X2),F = f()
    return F.subs({X1 : subX1, X2 : subX2 })

def fPrimeX1():
    (X1,X2), F = f()
    return ((X1,X2),diff(F,X1))

def fPrimeX1Sub(subX1, subX2=float('inf')):
    (X1,X2), FPX1 = fPrimeX1()
    if subX2 == float('inf'):
        return FPX1.subs(X1,subX1)
    else:
        return FPX1.subs({X1: subX1, X2: subX2})

def fPrimeX2():
    (X1,X2),F = f()
    return ((X1,X2),diff(F,X2))

def fPrimeX2Sub(subX2,subX1=float('inf')):
    (X1,X2), FPX2 = fPrimeX2()
    if subX1 == float("inf"):
        return FPX2.subs(X2,subX2)
    else:
        return FPX2.subs({X1: subX1, X2: subX2})

def gradientF():
    (X1,X2), FPX1 = fPrimeX1()
    (X1,X2), FPX2 = fPrimeX2()
    return ((X1,X2),(FPX1,FPX2))

def gradientFSub(subX1,subX2):
    # (X1,X2), FPX1 = fPrimeX1()
    # (X1,X2), FPX2 = fPrimeX2()
    return (fPrimeX1Sub(subX1=subX1, subX2=subX2), fPrimeX2Sub(subX1=subX1, subX2=subX2))

def gArmijoSub(alpha, x, d):
    return fSub(x[0] + alpha*d[0], x[1] + alpha*d[1])

def gPrimeArmijoSub(x ,d):
    vecFsub = gradientFSub(x[0], x[1])
    return vecFsub[0] * d[0] + vecFsub[1] * d[1]


def armijoConditionMax(x, d, c=0.2, alpha_init=1, scale=1.25):
    tolerance = 1e-10
    alpha = alpha_init
    ifCondition = gArmijoSub(alpha,x,d) <= (gArmijoSub(0,x,d) + c*alpha*gPrimeArmijoSub(x, d))
    #if ifcondition == false than alpha_init is less than alphaMax
    #if ifcondition == true than alpha_init is more than alphaMax

    whileCondition = abs(gArmijoSub(alpha,x,d) - (gArmijoSub(0,x,d) + c*alpha*gPrimeArmijoSub(x, d))) < tolerance
    if ifCondition == false:
        while not(abs(gArmijoSub(alpha,x,d) - (gArmijoSub(0,x,d) + c*alpha*gPrimeArmijoSub(x, d))) < tolerance):
            alpha *= scale
    else:
        scale = 1 - abs(scale - 1)
        while not(abs(gArmijoSub(alpha,x,d) - (gArmijoSub(0,x,d) + c*alpha*gPrimeArmijoSub(x, d))) < tolerance):
            alpha *= scale
        
    return alpha

def armijoconditionMin(x, d, c=0.4, alpha_init=1, scale=0.75):
    tolerance = 1e-5
    alpha = alpha_init
    ifCondition = gArmijoSub(2*alpha,x,d) > (gArmijoSub(0,x,d) + c*alpha*gPrimeArmijoSub(x, d))
    #if ifCondition == false than alpha_init is more than alphaMin
    #if ifCondition == true than alpha_init is less than alphaMin

    if ifCondition == false:
        while abs(gArmijoSub(2*alpha,x,d) - (gArmijoSub(0,x,d) + c*alpha*gPrimeArmijoSub(x, d))) >= tolerance:
            alpha *= scale
            print(alpha , abs(gArmijoSub(2*alpha,x,d) - (gArmijoSub(0,x,d) + c*alpha*gPrimeArmijoSub(x, d))))
    else:
        scale = 1 + abs(scale - 1)
        while abs(gArmijoSub(2*alpha,x,d) - (gArmijoSub(0,x,d) + c*alpha*gPrimeArmijoSub(x, d))) >= tolerance:
            alpha *= scale
            print(alpha , abs(gArmijoSub(2*alpha,x,d) - (gArmijoSub(0,x,d) + c*alpha*gPrimeArmijoSub(x, d))))
    return alpha


def firstStep(iteration, initGuess, preX=float('inf')):
    """first step initialization"""
    if iteration == 1:
        x = initGuess
        return x
    else:
        x = preX
        return x
    
# --------------------------------------<>------------------------------------------
def getCoordinateDirection(i):
    direction = [0,0]
    direction[i] += 1
    return direction

def vecScalerMult(landa, vec):
    return landa * vec[0], landa * vec[1]

def vecSum(vec1, vec2):
    return ((vec1[0] + vec2[0]), (vec1[1] + vec2[1]))

def secondStep(x):
    """for updating x"""
    minPoint = (0,0)
    minAmount = float('inf')

    for i in range(0,len(getCoordinateDirection(0))):
        print(f"i: {i}")
        d = getCoordinateDirection(i)
        print(f"d: {d}")
        alpha=(armijoconditionMin(x=x,d=d) + armijoConditionMax(x=x,d=d))/2
        print(f"alpha: {alpha}")

        constructedPoint = vecSum(x, vecScalerMult(alpha, d))
        constructedAmount = fSub(constructedPoint[0] , constructedPoint[1])
        if minAmount < constructedAmount:
            minPoint = constructedPoint
            minAmount = constructedAmount
    return minPoint

#------------------<>---------------------------
def terminate(ebsilon,preX,iteration,initGuess,x):
    """true == termination , false == continuation"""
    if iteration == 1:
        if abs(initGuess - x) > ebsilon:
            return false
        else:
            return true
    else:
        if abs(x - preX) > ebsilon:
            return false
        else:
            return true

def thirdStep(ebsilon,preX,iteration,initGuess,x):
    iteration += 1
    if terminate(ebsilon,preX,initGuess,x):
        return x
    else:
        return CCM(iteration=iteration, preX=x)


def CCM(initGuess=float('inf'), ebsilon=0.000001, iteration=1, preX=float('inf')):
    """Cyclic Coordinate Method (Armijo for step Length)"""

    print_box(iteration)

    print("." * 40)
    x = firstStep(iteration=iteration, initGuess=initGuess, preX=preX)
    print(f"firstStep\nx = {x}")
    print("." * 40)
    print()

    print("." * 40)
    x = secondStep(x)
    print(f"secondStep\nx = {x}")
    print("." * 40)
    
    return thirdStep(ebsilon=ebsilon, preX=preX, iteration=iteration, initGuess=initGuess, x=x)



print(f"result: {CCM(initGuess=(0,3))}")
# print(f"f: {f()}")
# print(f"fprimex1(): {fPrimeX1()}")
# print(f"fprimsub: {fPrimeX1Sub(1)}")
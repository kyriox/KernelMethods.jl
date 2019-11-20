# Copyright 2019 Jose Ortiz-Bejar 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#module Nets
export Net, fft_sampling, kmeans_sampling, density_sampling, random_sampling, gen_features, KMS, f1m
#import KernelMethods.Kernels: sigmoid, gaussian, linear, cauchy, poly, quadratic
using KernelMethods: accuracy, recall#, f1
#import KernelMethods.Supervised: NearNeighborClassifier, optimize!, predict_one, predict_one_proba
using KernelMethods: NearNeighborClassifier, optimize!, predict_one, predict_one_proba
using SimilaritySearch: KnnResult, l2_distance, normalize, squared_l2_distance,  hamming_distance, cosine_distance, angle_distance
using PyCall
import PyCall: PyObject, pyimport
using StatsBase
using Random
using Distributed

#nb=pyimport("sklearn.naive_bayes")
#nn=pyimport("sklearn.neighbors")
#ms=pyimport("sklearn.model_selection")
scipy=pyimport("scipy.stats")
#@pyimport  scipy.stats as scipy
#using DataStructures
# Samplinng Net structure
mutable struct Net
    data::Vector{Vector{Float64}}
    labels::Vector
    #udata::Vector{Vector{Float64}} #unlabeled data
    udata::Vector{Vector{Float64}} #Vector of unlabeled data
    references::Vector{Int32}
    partitions::Vector{Int32}
    centers::Vector#{Vector{Float64}}
    centroids::Vector#{Vector{Float64}}
    dists::Vector{Float64}
    csigmas::Vector{Float64}
    sigmas::Dict{Int,Float64}
    stats::Dict{String,Float64}
    reftype::Symbol
    distance::Symbol
    kernel
    clftype::Symbol #:space or prototypes
    clabels::Vector{Int}
    #sparse::Bool
    #vbows::Vector{BOW}
end


Net(data,labels;udata=[])::Net=Net(data,labels,udata,[],[],[],[],[],[],Dict(),Dict(),
                                    :centroids,:squared_l2_distance,gaussian,:space,[])

##  Kernel Fucntions

function linear(xo,xm,distance;sigma=1)::Float64
    d=distance(xo,xm)
    return d #<=1e-6 ? 1.0 : sigma/d
end

poly(xo,xm,distance;sigma=1,degree=2)=(sigma*distance(xo,xm)+1)^degree

quadratic(xo,xm,distance;sigma=1)=1-distance(xo,xm)^2/(distance(xo,xm)^2+sigma)

maxk(xo,xm,distance;sigma=1) = distance(xo,xm) > sigma ? 1.0 : 0.0

function gaussian(xo,xm,distance; sigma=1)::Float64
    d=distance(xo,xm)
    if d<=1e-6
        return 1.0
    end
    exp(-d^2/(2*sigma^2))
end

sigmoid(xo,xm,distance; sigma=1)=1/(1+exp(distance(xo,xm)-sigma))

function cauchy(xo,xm,distance; sigma=1)
    d=distance(xo,xm)
    if d<=1e-6 
        return 1.0
    end
    return 1/(1+d^2/sigma^2)
end

# AUC score
function auc(y,yp)::Float64
    return 2*metrics.roc_auc_score(y, yp)-1
end
# Angle distance of normalize vectors
function angle(a,b)
    an=normalize(a)
    bn=normalize(b)
    angle_distance(an,bn)
end

function f1m(y,yp)::Float64
    metrics=pyimport("sklearn.metrics")
    return metrics.f1_score(y, yp, average="macro")
end

function pearson(y,yp)::Float64
    return scipy.pearsonr(y, yp)[1]
end

function rmse(y,yp)::Float64
    yp=[x>0 ? x : 0 for x in yp]
    aux=log.(yp+1).-log.(y+1)
    sqrt(sum(aux.*aux)/length(y))
end

## Find the next farthest element
function maxmin(data,centers,ind,index,distance,partitions)::Tuple{Int64,Float64}
    c=last(centers)
    if length(index)==0
        for i in ind
            if i!=c
                push!(index,i,Inf)
            end
        end
    end
    nindex=KnnResult(length(index))
    for fn in index
        dist=eval(distance)(data[fn.objID],data[c])
        #push!(lK[fn.objID],dist)
        #@show dist, fn.dist , distance,fn.objID
        #@show data[fn.objID]
        #@show data[c]
        dist = (dist<fn.dist) ?  dist : fn.dist
        partitions[fn.objID] = (dist<fn.dist) ? c : partitions[fn.objID]
        if fn.objID!=c
            if typeof(dist)==Symbol
                @show dist
            end
            push!(nindex,fn.objID,convert(Float64, dist))
        end
    end
    index.k=nindex.k
    index.pool=nindex.pool
    fn=pop!(index)
    return fn.objID,fn.dist
end

## Centroids  using only non zero values
function mean_non_zero(data::Vector{T})::Vector{Float64} where T
    n=length(data[1])
    vmean=Vector{Float64}(undef,n)
    for i in 1:n
        tmp=[x[i] for x in data if x[i]>0]
        vmean[i]= length(tmp)>0 ? mean(tmp) : 0
    end
    vmean
end

function cmean(data)
    n=length(data)
    #if typeof(data[1])==TextSearch.BOW
    #    total=deepcopy(data[1])
    #    for vbow in data[2:end]
    #        total=total+vbow
    #    end
    #    return total
    #else
    return mean(data)
    #end
end

# get a set of Centroids
function get_centroids(N)
    centers=[j for j in Set(N.partitions)]
    sort!(centers)
    #if N.sparse
    #    centroids=Vector{BOW}(length(centers))
    #    data= N.vbows[:]
    #else
    centroids=Vector{Vector{Float64}}(undef,length(centers))
    data = N.data[:]
    #end
    [push!(data,ud) for ud in N.udata];
    for (ic,c) in enumerate(centers)
        ind=[i for (i,v) in enumerate(N.partitions) if v==c]
        #if N.distance!=:cosine 
        ##centroids[ic]=mean(N.data[ind])
        centroids[ic]=cmean(data[ind])
        #else
        #centroids[ic]=mean_non_zero(N.data[ind])
        #end
    end
    return centroids
end

function to_column(data)
    X=hcat(data...)
    X=[X[i,:] for i in  1:size(X)[1]]
    return X
end

# Epsilon Network using farthest first traversal Algorithm
function fft_sampling(N,num_of_centers::Int; distance=:squared_l2_distance, axis=1,
    per_class=false,reftype=:centroids, kernel=linear,test_set=false,)
    N.distance=distance
    N.kernel=kernel
    #@show num_of_centers distance  axis N.sparse  reftype per_class Symbol(kernel)
    ##n=length(N.data)
    yc=[]
    #if N.sparse && axis > 0
    #    data=N.vbows
    #    n=length(N.vbows)+length(N.udata)
    #else
    data=N.data
    n=length(N.data)+length(N.udata)
    #end
    ######### Real Inductive ######
    if test_set && length(N.udata)>0 && !per_class
        data=[]
        n=length(N.udata)
    end
    ##############################
    # @show n,N.sparse
    partitions=[0 for i in 1:n]
    gcenters,dists,sigmas=Vector{Int}(undef,0),Vector{Float64}(undef,num_of_centers-1),Dict{Int,Float64}()
    indices=[[i for i in  1:n]]
    #indices=[[i for (i,j) in enumerate(N.labels) if j==l] for l in [1]]
    #@show length(indices)
    L=sort([i for i in Set(N.labels)])
    if per_class
        indices=[[i for (i,j) in enumerate(N.labels) if j==l] for l in L]
        #@show length(indices), "XXXXXXXXXXXXXXXX"
    end
    fdata=data[:]
    if !per_class
        [push!(fdata,ud) for ud in N.udata];
    end
    si=1
    for ind in indices
        centers=Vector{Int}(undef,0)
        s=rand(1:length(ind))
        push!(centers,ind[s])
        index=KnnResult(length(ind))
        partitions[ind[s]]=ind[s]
        k=1
        push!(yc,L[si])
        while  k<=num_of_centers-1 && k<=length(ind)
            ##fnid,d=maxmin(N.data,centers,ind,index,distance,partitions)
            fnid,d=maxmin(fdata,centers,ind,index,distance,partitions)
            push!(centers,fnid)
            dists[k]=d
            partitions[fnid]=fnid
            k+=1
            push!(yc,L[si])
        end
        N.sigmas[L[si]]=minimum(dists)
        si=si+1
        gcenters=vcat(gcenters,centers)
    end
    assign(fdata,fdata[gcenters],partitions;distance=N.distance)
    N.references,N.partitions,N.dists,N.sigmas=gcenters,partitions,dists,sigmas
    ##N.csigmas,N.stats=get_csigmas(N.data,N.centroids,N.partitions,distance=N.distance)
    N.csigmas,N.stats=get_csigmas(fdata,N.centroids,N.partitions,distance=N.distance)
    #N.sigmas[0]=maximum(N.csigmas)
    ##N.centers,N.centroids=N.data[gcenters],get_centroids(N)
    N.centers,N.centroids=fdata[gcenters],get_centroids(N)
    N.reftype=reftype
    N.clabels=yc
end


# KMeans ++ seeding Algorithm 

function kmpp(N,num_of_centers::Int)::Vector{Int}
    fdata=N.data[:]
    #if !per_class
    [push!(fdata,ud) for ud in N.udata];
    #end
    n=length(fdata)
    #partitions=[0 for i in 1:n]
    #n=length(N.data)
    s=rand(1:n)
    centers,d=Vector{Int}(undef,num_of_centers),squared_l2_distance
    centers[1]=s
    #D=[d(N.data[j],N.data[s]) for j in 1:n]
    D=[d(fdata[j],fdata[s]) for j in 1:n]
    for i in 1:num_of_centers-1
        cp=cumsum(D/sum(D))
        r=rand()
        sl=[j for j in 1:length(cp) if cp[j]>=r]
        s=sl[1]
        centers[i+1]=s
        for j in 1:n
            dist=d(fdata[j],fdata[s])
            # dist=d(N.data[j],N.data[s])
            if dist<D[j]
                D[j]=dist
            end
        end
    end
    #assign(fdata,fdata[centers],partitions;distance=N.distance)
    centers
end

#Assign Elementes to thier  nearest centroid

function assign(data,centroids,partitions;distance=:squared_l2_distance)
    d=distance
    #@show centroids
    #@show data
    for i in 1:length(data)
        dd=[eval(d)(data[i],c) for c in centroids]
        #@show dd
        partitions[i]=sortperm([eval(d)(data[i],c) for c in centroids])[1]
    end
end

function assignc(data,centroids,partitiontails;distance=:squared_l2_distance)
    d=distance
    for i in 1:length(data)
        partitions[i]=sortperm([eval(d)(data[i],c) for c in centroids])[1]
    end
end

#Distances for each element to its nearest cluster centroid

function get_distances(data,centroids,partitions;distance=:squared_l2_distance)::Vector{Float64}
    dists=Vector{Float64}(undef,length(centroids))
    for i in 1:length(centroids)
        ind=[j for (j,l) in enumerate(partitions) if l==i]
        if length(ind)>0
            X=data[ind]
            dd=[eval(distance)(centroids[i],x) for x in X]
            dists[i]=maximum(dd)
        end
    end
    sort!(dists)
    return dists
end

#Calculated the sigma for each ball

function get_csigmas(data,centroids,partitions;distance=:squared_l2_distance)::Tuple{Vector{Float64},Dict{String,Float64}}
    stats=Dict("SSE"=>0.0,"BSS"=>0.0)
    refs=[j for j in Set(partitions)]
    sort!(refs)
    df=eval(distance)
    if distance==:squared_l2_distance
        df=eval(:l2_distance)
    end
    csigmas=Vector{Float64}(undef,length(refs))
    for (ii,i) in enumerate(refs)
        ind=[j for (j,l) in enumerate(partitions) if l==i]
        #if length(ind)>0
        X=data[ind]
        dd=[df(data[i],x) for x in X]
        csigmas[ii]=max(0,maximum(dd))
        #stats["SSE"]+=sum(dd)
        #stats["BSS"]+=length(X)*(sum(mean(X)-m))^2
        #end
    end
    return csigmas,stats
end

#Feature generator using kmeans centroids

function kmeans_sampling(N,num_of_centers::Int; max_iter=1000,kernel=linear,distance=:squared_l2_distance,reftype=:centroids, per_class=false,test_set=false)
    #n=length(N.data)
    fdata=N.data[:]
    #if !per_class
    [push!(fdata,ud) for ud in N.udata];
    #end
    n=length(fdata)
    #lK,partitions=[[] for i in 1:n],[0 for i in 1:n],[0 for i in 1:n]
    partitions=[0 for i in 1:n]
    dists=Vector{Float64}
    init=kmpp(N,num_of_centers)
    centroids=fdata[init]#N.data[init]
    i,aux=1,Vector{Float64}(undef,length(centroids))
    while centroids != aux && i<max_iter
        i=i+1
        aux = centroids
        assign(fdata,centroids,partitions)
        #assign(N.data,centroids,partitions)
        N.partitions=partitions
        centroids=get_centroids(N)
    end
    N.distance=distance
    dists=get_distances(fdata,centroids,partitions,distance=N.distance)
    N.partitions,N.dists=partitions,dists
    N.centroids,N.sigmas[0]=centroids,maximum(N.dists)
    #N.csigmas,N.stats=get_csigmas(N.data,N.centroids,N.partitions,distance=N.distance)
    N.centers=init
    N.csigmas,N.stats=get_csigmas(fdata,N.centroids,N.partitions,distance=N.distance)
    N.sigmas[0]=maximum(N.csigmas)
    N.reftype=:centroids
    N.kernel=kernel
end


#References selection using naive algoritmh for density net

function density_sampling(N,num_of_elements::Int64; distance=:squared_l2_distance,kernel=linear,reftype=:centroids,per_class=false, test_set=false)
    N.distance=distance
    #if N.sparse
    #    data=N.vbows
    #else
    data=N.data
    #end
    fdata=data[:]
    if !per_class
        [push!(fdata,ud) for ud in N.udata];
    end
    n,d,k=length(fdata),distance,num_of_elements
    k=trunc(Int,n/k)
    partitions,references=[0 for i in 1:n],Vector{Int}(undef,0)
    pk=1
    dists,sigmas=Vector{Float64},Dict{Int,Float64}()
    while 0 in partitions
        pending=[j for (j,v) in enumerate(partitions) if partitions[j]==0]
        s=rand(pending)
        partitions[s]=s
        pending=[j for (j,v) in enumerate(partitions) if partitions[j]==0]
        push!(references,s)
        pc=sortperm([eval(d)(fdata[j],fdata[s]) for j in pending])
        if length(pc)>=k
            partitions[pending[pc[1:k]]]=[s for j in 1:k]
        else
            partitions[pending[pc]]=[s for j in 1:length(pc)]
        end
    end
    N.references,N.partitions=references,partitions
    N.centers,N.centroids=fdata[references],get_centroids(N)
    N.csigmas,N.stats=get_csigmas(fdata,N.centroids,N.partitions,distance=N.distance)
    N.sigmas[0]=maximum(N.csigmas)
    N.reftype=:centroids
    N.distance=distance
    N.kernel=kernel
end

# Select references randomly
function random_sampling(N,num_of_elements::Int64; distance=:squared_l2_distance,kernel=linear,reftype=:centroids, per_class=false,
    test_set=false)
    N.distance=distance
    #if N.sparse
    #    data=N.vbows 
    #else
    data=N.data
    #end
    fdata=data[:] 
    [push!(fdata,ud) for ud in N.udata];
    n,d,k=length(fdata),distance,num_of_elements
    partitions=[0 for i in 1:length(fdata)]
    pk=1
    dists,sigmas=Vector{Float64},Dict{Int,Float64}()
    references=Random.randperm(n)[1:num_of_elements]
    assign(fdata,fdata[references],partitions,distance=distance)
    N.references,N.partitions=references,partitions
    N.centers,N.centroids=fdata[references],get_centroids(N)
    N.csigmas,N.stats=get_csigmas(fdata,N.centroids,N.partitions,distance=N.distance)
    N.sigmas[0]=maximum(N.csigmas)
    N.reftype=:centroids
    N.distance=distance
    N.kernel=kernel
end

#Find Inverse Homegeinity Index
function IHomogeneity(N)
    G=Set(N.partitions)
    k=length(G)
    h=0.0
    for g in G
        ind=[i for (i,l) in enumerate(N.partitions) if l==g]
        gl=N.labels[ind]
        nc=length(gl)
        mc=first(sort([ v for (k,v) in countmap(gl)],rev=true))
        h+=1/nc*(mc/length(gl)) 
    end 
    k-h
end

#Generates feature espace using cluster centroids or centers
function gen_features(Xo,N)::Vector{Vector{Float64}}
    n=length(Xo)
    sigmas,Xr=N.csigmas,Vector{Vector{Float64}}(undef,n)
    #@show length(N.centers), length(N.centroids)
    Xm = N.reftype==:centroids || length(N.centers)==0 ? N.centroids : N.centers 
    nf=length(Xm)
    kernel=N.kernel
    #@show Xm
    #@show N.partitions, N.reftype, N.centers
    for i in 1:n
        xd=Vector{Float64}(undef,nf)
        for j in 1:nf  
            #@show i,j,length(Xo),length(Xm),length(sigmas),sigmas
            #@show Xo[i],Xm[j],sigmas[j]
            #xd[j]=kernel(Xo[i],Xm[j],sigma=sigmas[j],distance=eval(N.distance))
            dd=N.distance
            if N.distance==:squared_l2_distance
                dd=:l2_distance 
            end
            xd[j]=kernel(Xo[i],Xm[j],eval(dd),sigma=sigmas[j])
        end
        xd[isnan.(xd)].=0.0
        if typeof(xd)==Symbol
            @show "============XXXXXXXXXXXXXXXX======================", xd, i
        end
        Xr[i]=xd
    end
    Xr
end



function KFoldsTrain(N;op_function=recall,folds=3,runs=0,trainratio=0.7,testratio=0.3)
    #data= N.sparse ? N.vbows : N.data
    data=N.data
    X=gen_features(data,N)
    y=N.labels 
    ms=pyimport("sklearn.model_selection")
    skf=ms.StratifiedKFold(n_splits=folds)
    skf[:get_n_splits](X,y)
    clfs=genCl()
    opval,cltr,disttr,wtr=-1,"NaiveBayes","ND","NA"
    clfr=clfs[1][1]
    for (clf,clt,distt,wt) in clfs
        y_pred=[]
        yr=[]
        for (ei,vi) in skf[:split](X,y)
            ei,vi=ei+1,vi+1
            xt,xv=X[ei],X[vi]
            yt,yv=y[ei],y[vi]
            #clf[:fit](xt,yt)
            clf.fit(xt,yt)
            #y_pred=clf[:predict](xv)
            yr=vcat(yr,yv)
            #y_pred=vcat(y_pred,clf[:predict](xv)[:,1])
            y_pred=vcat(y_pred,clf.predict(xv)[:,1])
        end
        opv=op_function(yr,y_pred)
        if opv>opval
            clfr,cltr,disttr,wtr=clf,clt,distt,wt
            opval=opv
        end
    end
    #@show  opval, cltr, disttr, N.distance, trainratio,length(N.centroids)
    #clfr[:fit](X,y)
    clfr.fit(X,y)
    return (clfr,opval,cltr,disttr,wtr)
end




function MonteCarloTrain(N;op_function=recall,runs=3,trainratio=0.7,testratio=0.3,folds=0)
    #data= N.sparse ? N.vbows : N.data
    data= N.data
    X=gen_features(data,N)
    y=N.labels    
    testratio = testratio>0.0 ? testratio : 1.0-trainratio 
    clfs=genCl()
    opval,cltr,disttr,wtr=-1,"MLP","ND","NA"
    clfr=clfs[1][1]
    res=[0 for  clf in clfs]
    for i  in 1:runs
        j=1
        for (clf,clt,distt,wt) in clfs
            xt,xv,yt,yv=ms.train_test_split(X,y,train_size=trainratio,test_size=testratio)
            #clf[:fit](xt,yt)
            clf.fit(xt,yt)
            #y_pred=clf[:predict](xv)[:,1]
            y_pred=clf.predict(xv)[:,1]
            opv=op_function(yv,y_pred)
            res[j]=res[j]+opv
            j+=1
        end
    end
    m=sortperm(res)[end]
    clfr,cltr,disttr,wtr=clfs[m]
    opval=res[m]/runs
    #clfr[:fit](X,y)
    clfr.fit(X,y)
    return (clfr,opval,cltr,disttr,wtr)
end


function predict_test(xt,y,xv,desc,cl)
    y_pred=[] 
    nb=pyimport("sklearn.naive_bayes")         
    if contains(desc,"KNN")
        #@ typeof(xt), typeof(y), typeof(cl.X.dist), Symbol(cl.X.dist)
        cln=NearNeighborClassifier(xt,y,cl.X.dist,cl.k,cl.weight)  
        y_pred=[predict_one_proba(cln,x) for x in xv]
    elseif  contains(desc,"SVC")
        y_tmp=cln[:decision_function](xv)
        y_pred=[[0,1/(1+exp(-yp))] for yp in y_tmp]
    else
        cln=nb.GaussianNB()
        #cln[:fit](xt,y)
        cln.fit(xt,y)
        #y_pred=cln[:predict_proba](xv) 
        y_pred=cln.predict_proba(xv) 
    end 
    y_pred
end

function fisher(N)::Float64
    data,labels=gen_features(N.data,N),N.labels
    classes=Set(labels)
    n=length(data)
    m=length(data[1])
    nc=length(classes)
    cl=Vector{Vector{Int64}}(undef,n)
    for (i,class) in enumerate(classes)
        cl[i]=[j for (j,l) in enumerate(labels) if l==class]
    end
    x1,x2=hcat(data[cl[1]]...)',hcat(data[cl[2]]...)'
    m1,m2=mean(x1,1),mean(x2,1)
    s1,s2=std(x1,1),std(x2,1)
    sum(((m1-m2).^2)/((s1+s2).^2))
end

function genCl()
    K=[1,5,11,21]
    D=["cosine","euclidean"]
    W=["uniform","distance"]
    clfs=[]
    nn=pyimport("sklearn.neighbors")
    nb=pyimport("sklearn.naive_bayes")
    lm=pyimport("sklearn.linear_model")
    svm=pyimport("sklearn.svm")
    for d in D
        for w in W
            for k in K
                if k==1 && w=="distance"
                    continue 
                else
                    clf=nn.KNeighborsClassifier(n_neighbors=k,metric=d,weights=w)
                    #@show clf
                    push!(clfs,(clf,"NN$k",d,w))
                end
            end
        end
    end
    clf_list=[clfs[Random.randperm(length(clfs))][1], (nb.GaussianNB(),"NaiveBayes","ND","NA")]
    push!(clf_list,(lm.RidgeClassifier(),"Ridge","ND","NA"))
    push!(clf_list,(lm.LogisticRegression(),"Logistic","ND","NA"))
    push!(clf_list,(svm.LinearSVC(),"SVM","ND","NA"))
    return clf_list
    #return clf_list[Random.randperm(length(clf_list))][1]
end


function genGrid(nets=[:fft_sampling,:kmeans_sampling,:density_sampling,:random_sampling];K=[4,8,16,32],
    trainings=[:inductive],
    kernels=[:gaussian,:sigmoid,:linear,:cauchy],
    reftypes=[:centers,:centroids],
    distances=[:angle,:squared_l2_distance],
    distancesk=[:angle,:squared_l2_distance],
    #trainratios=[1],
    sample_size=128)
    #clfs=genCl()
    if length(nets)==1 && nets[1]==:kmeans_sampling
        reftypes=[:centroids]
        distancesk=[:squared_l2_distance] 
    end
    #space=[(k=k,kernel=kernel,reftype=reftype,distancek=dc,nettype=net,training=training,cl=genCl()) 
    space=[(k=k,kernel=kernel,reftype=reftype,distancek=dc,nettype=net,training=training,cl=cl) 
    for k in K  for kernel in kernels 
    for reftype in reftypes for dc in distancesk for net in nets 
    for training in trainings 
    for cl in genCl() 
    if  net!=:kmeans_sampling || reftype!=:centers || (net==:kmeans_sampling && dc==:squared_l2_distance)]
    #if  !(net==:kmeans_sampling  && dc==:angle)]
    #sz = sample_size%2==1 ? trunc(Int,sample_size/2)+1 : trunc(Int,sample_size/2)
    if length(space)>sample_size && sample_size!=-1
        space=space[Random.randperm(length(space))[1:sample_size]]
    end
    space
end

function inductive(Xe,Ye,k,nettype,kernel,distancek,reftype,classifier;
    distances=[:angle,:squared_l2_distance], 
    op_function=:recall,folds=3,udata=[], per_class=false, 
    test_set=false)
    ms = pyimport("sklearn.model_selection")
    skf=ms.StratifiedKFold(n_splits=folds,shuffle=true)
    skf.get_n_splits(Xe,Ye)
    clf,clt,distt,wt=classifier
    y_pred=[]
    yr=[]
    #@info  "Starting KFOLDS", k,nettype,kernel,distancek,reftype
    for (ei,vi) in skf.split(Xe,Ye)
        ei,vi=ei.+1,vi.+1
        xt,xv=Xe[ei],Xe[vi]
        yt,yv=Ye[ei],Ye[vi]
        N=Net(xt,yt)
        N.data=xt
        N.labels=yt
        eval(nettype)(N,k,kernel=eval(kernel),distance=distancek,reftype=reftype,per_class=per_class)
        Xt=gen_features(xt,N)
        Xv=gen_features(xv,N)
        clf.fit(Xt,yt)
        yr=vcat(yr,yv)
        y_pred=vcat(y_pred,clf.predict(Xv)[:,1])
    end
    #@info  "Finished KFOLDS", k,nettype,kernel,distancek,reftype
    #@info  "Starting Final training =========", k,nettype,kernel,distancek,reftype
    opval=eval(op_function)(yr,y_pred)
    #@info  "Building final NET XXXXXXXXXXXXXXXX"
    Nf=Net(Xe,Ye)
    eval(nettype)(Nf,k,kernel=eval(kernel),distance=distancek,reftype=reftype,per_class=per_class)
    #@info  "final NET finished XXXXXXXXXXXXXXXX"
    #@info  "Generating features for full data"
    X=gen_features(Xe,Nf)
    #@info  "features for full data generated"
    #@info  "Fitting final clasifier"
    clf.fit(X,Ye)
    #@info  "final clasifier fitted =============="
    trainratio=1
    traintype="KFolds"
    key="$nettype/$kernel/$distancek/$k/$clt/$reftype/$trainratio/inductive/$distt/$wt/$traintype"
    return (clf,Nf),(opval,key)
    #return (clfr,opval,cltr,disttr,wtr)
end

function eval_conf(args)
    c,op_function,Xe,Ye,per_class,test_set,folds,udata,debug=args
    #@info length(c)
    #@info  "Configuration Inited", c.k, c.kernel, c.reftype,c.distancek,c.nettype,c.training,length(c.cl) 
    #@info op_function,length(Xe),length(Ye),per_class,test_set,folds,udata
    #@info c,op_function,Xe,Ye,per_class,test_set,folds,udata
    (cli,neti),(opvali,ckeyi) = eval(c.training)(Xe,Ye,c.k,c.nettype,c.kernel,c.distancek,c.reftype,
    c.cl; folds=folds,udata=udata, op_function=op_function, per_class=per_class,test_set=test_set)
    #@info "Configuration Evaluated", c.k, c.kernel, c.reftype,c.distancek,c.nettype,c.training,length(c.cl) 
    if debug
        @show "#########", opvali, ckeyi
    end
    (cl=cli, net=neti, opval=opvali, ckey=ckeyi)
end

function KMS(Xe,Ye; op_function=:recall,top_k=15,folds=3,per_class=false, udata=[],
    nets=[:fft_sampling,:kmeans_sampling,:density_sampling,:random_sampling],
    K=[4,8,16,32,64],distances=[:angle,:squared_l2_distance],
    distancesk=[:angle,:squared_l2_distance],sample_size=128,
    kernels=[:gaussian,:linear,:cauchy,:sigmoid],test_set=false, debug=false)
    #DNNC=Dict()
    space_temp=genGrid(nets,K=K,kernels=kernels,distancesk=distancesk,sample_size=sample_size,distances=distances)
    space=[(conf,op_function,Xe,Ye,per_class,test_set,folds,udata,debug) for conf in space_temp]
    res=map(eval_conf, space)
    sort!(res, by=x->x.opval, rev=true)
    res[1:top_k]
    #(cli,neti),(opvali,ckeyi)=
    #Top=Vector{Tuple{Float64,String}}(undef, 0)
    #for (k,kernel,reftype,distancek,nettype,training,clfc) in space
        #cln,dkn,cldn=clfc[2],clfc[3],clfc[4]
    #    (cli,neti),(opvali,ckeyi)=eval(training)(Xe,Ye,k,nettype,kernel,distancek,reftype,
    #    clfc; folds=folds,udata=udata, op_function=op_function, per_class=per_class,test_set=test_set)
    #    push!(Top,(opvali,ckeyi))
    #    if debug
    #        @show (debug, opvali,ckeyi)
    #    end
    #    DNNC[ckeyi]=(cli,neti)
    #end   
    #for (k,kernel,reftype,distancek,nettype,training) in space
    #    clf_list=genCl()
    #    if sample_size!=-1
    #        clf_list=[clf_list[Random.randperm(length(clf_list))][1]]
    #    end
    #    #push!(clf_list,(nb.GaussianNB(),"NaiveBayes","ND","NA"))
    #    #@show(length(space)*length(clf_list))
    #    cl,net,opval,ckey=nothing,nothing,0,""
    #    clf_list
    #    for clfc in clf_list
    #       cln,dkn,cldn=clfc[2],clfc[3],clfc[4]
    #        (cli,neti),(opvali,ckeyi)=eval(training)(Xe,Ye,k,nettype,kernel,distancek,reftype,clfc;
    #        folds=folds,udata=udata, 
    #        op_function=op_function, per_class=per_class,test_set=test_set)
    #        if opvali>opval
    #            (cl,net),(opval,ckey)=(cli,neti),(opvali,ckeyi)
    #        end
    #    end
    #    push!(Top,(opval,ckey))
    #    if debug
    #        @show (debug, opval,ckey)
    #    end
    #    DNNC[ckey]=(cl,net)
    #end
    #sort!(Top,rev=true)
    #Top=Top[1:top_k]
    #[(DNNC[top[2]],top[1],top[2]) for top in Top]
end


#Ensemble based on configuration
function ensemble_cfft(knc,k::Int64=7;testratio=0.4,distance=:hamming_distance)::Vector{Tuple{Tuple{Any,Net},Float64,String}}
    (cl,n),opv,desc=knc[1] 
    data=Vector{Vector{String}}(undef, length(knc))
    for i in 1:length(knc)
        kc,opv,desc=knc[i]
        v=split(desc,"/")[1:6]
        data[i]=v
    end
    ind=[i for (i,x) in enumerate(data)]
    partitions,centers,index=[0 for x in ind],[1],KnnResult(k)
    while length(centers)<k && length(centers)<=length(ind) 
        oid,dist=maxmin(data,centers,ind,index,distance,partitions)
        push!(centers,oid)  
    end
    knc[centers]
end

#New features based ensemble

function ensemble_pfft(knc,k::Int64=7;trainratio=0.7,distance=:hamming_distance)::Vector{Tuple{Tuple{Any,Net},Float64,String}}
    (cl,N),opv,desc=knc[1] 
    n=length(N.data)
    tn=Int(trunc(n*trainratio));
    data=Vector{Vector{Float64}}(undef, length(knc))
    perm=Random.randperm(n)
    ti,vi=perm[1:tn],perm[tn+1:n] 
    for i in 1:length(knc)
        (cl,N),opv,desc=knc[i]
        xv=gen_features(N.data[vi],N)
        xt=gen_features(N.data[ti],N)
        y=N.labels[ti]
        data[i]=vcat(predict_test(xt,y,xv,desc,cl)...)
    end
    ind=[i for (i,x) in enumerate(data)]
    partitions,centers,index=[0 for x in ind],[1],KnnResult(k)
    while length(centers)<k && length(centers)<=length(ind) 
        oid,dist=maxmin(data,centers,ind,index,distance,partitions)
        push!(centers,oid)  
    end
    knc[centers]
end

function predict(knc,X;ensemble_k=1)
    y_t=Vector{Int}(undef, 0)
    #@show length(knc)
    for i in 1:ensemble_k
        #kc,opv,desc=knc[i].cl,knc[i].opval, knc[i].ckey
        #cl,N=kc
        cl,N,opv,desc=knc[i].cl,knc[i].net,knc[i].opval, knc[i].ckey
        xv=gen_features(X,N)       
        #if contains(desc,"KNN")  
        #    y_i=[predict_one(cl,x)[1] for x in xv]
        #else
        #y_i=cl[:predict](xv) 
        y_i=cl.predict(xv) 
        #end 
        y_t = length(y_t)>0 ? hcat(y_t,y_i) : hcat(y_i)
    end
    if length(y_t[:,1])>1
        y_pred=Vector{Int}(undef, length(X))
        for i in 1:length(y_t[:,1])
            y_r=y_t[i,:]
            y_pred[i]=last(sort([(count(x->x==k,y_r),k) for k in unique(y_r)]))[2]
        end
        y_pred
    else
        y_t[:,1]
    end
end

function predict_proba(knc,X):Vector{Vector{Float64}}
    kc,opv,desc=knc
    cl,N=kc
    xv=gen_features(X,N)
    if contains(desc,"KNN")  
        y_pred=[predict_one_proba(cl,x) for x in xv]
    elseif  contains(desc,"SVC")
        y_tmp=cl[:decision_function](xv)
        y_pred=[[0,1/(1+exp(-yp))] for yp in y_tmp]
    else
        #yp=cl[:predict_proba](xv)   
        yp=cl.predict_proba(xv)    
        y_pred=[yp[i,:] for i in 1:length(yp[:,1])] 
    end
    y_pred
end


function predict_two(knc,X;ensemble_k=5)::Vector{Int64}
    labels,y_pred,Xt,Xv=[],[],[],[]
    for i in 1:ensemble_k
        kc,opv,desc=knc[i]
        cl,N=kc
        labels=N.labels
        xv=predict_proba(knc[i],X)
        xt=predict_proba(knc[i],N.data)
        #@show xv
        Xv=length(Xv) == 0 ? xv : [vcat(a,b) for (a,b) in zip(Xv,xv)]
        Xt=length(Xt) == 0 ? xt : [vcat(a,b) for (a,b) in zip(Xt,xt)]
    end
    cl,(opval,desc)=transductive_pred(Xt,labels)
    if contains(desc,"KNN")  
        y_pred=[predict_one(cl,x)[1] for x in Xv]
    else
        #y_pred=cl[:predict](Xv)       
        y_pred=cl.predict(Xv)
    end
    y_pred
end


function predict_proba_ensemble(knc,X;ensemble_k=5)::Vector{Float64}
    labels,y_pred,Xt=[],[],[]
    for i in 1:ensemble_k
        kc,opv,desc=knc[i]
        cl,N=kc
        labels=N.labels
        xt=predict([knc[i]],X)
        Xt=length(Xt) == 0 ? xt : hcat(Xt,xt)
    end 
    #@show length(Xt[1])
    vcat(sum(Xt,2)/ensemble_k...)
end

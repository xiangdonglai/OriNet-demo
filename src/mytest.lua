require 'paths'
require 'torch'
require 'xlua'
require 'json'
paths.dofile('myutils.lua')
paths.dofile('ref.lua')
model = torch.load(opt.loadModel)

seqName = 'dslr_dance1'
ntest = 360
avglen = {117.80702013, 487.78421339, 373.71816181, 117.80702013,
       487.96079204, 373.71755575, 223.44787073, 250.85571875,
      85.11261748, 170.2761693 , 143.43147138, 299.4442099 ,
       227.34127406, 146.2194657 , 299.44218319, 227.34363571}

w = image.display(image.lena())
model:evaluate()
for idx = 1,ntest do
    xlua.progress(idx, ntest)
    inp = loadInTheWild(seqName, idx)
    image.display{image=inp, win=w}

    pred = model:forward(inp:cuda())
    orien = pred[9][{1,{},{},{}}]:float():clone()
    pred2D = getPreds(pred[10]:float():clone())
    pred2D = pred2D[{1,{},{}}]
    res = 64
    skel3D = torch.zeros(17,3)

    for id = 1,#dataset.skeletonRef do    
        idx1 = dataset.skeletonRef[id][1]
        idx2 = dataset.skeletonRef[id][2]
    
        jnt1 = pred2D[idx1]
        jnt2 = pred2D[idx2]
    
        l = torch.norm(jnt2 - jnt1)
        v = (jnt2 - jnt1) / l
        vp = torch.Tensor({v[2],-v[1]})
        tmp = torch.zeros(3)
        left = torch.cmin(jnt1,jnt2)
        right = torch.cmax(jnt1,jnt2)
        for i = left[2],right[2] do
            for j = left[1],right[1] do
                pos = torch.Tensor({j-jnt1[1],i-jnt1[2]})
                pro1 = v:dot(pos)
                pro2 = vp:dot(pos)
                tnorm = torch.norm(orien[{{3*id-2,3*id},i,j}]) 
                if pro1 >= 0 and pro1 <= l and torch.abs(pro2) <= 1 and tnorm > 0.1 then
                    tmp = tmp + orien[{{3*id-2,3*id},i,j}] / tnorm
                end
            end
        end
        if torch.norm(tmp) < 0.1 then
            for ii = 1,3 do
                t1 = torch.max(orien[{id*3-3+ii,{},{}}])
                t2 = torch.min(orien[{id*3-3+ii,{},{}}])
                if t1 + t2 < 0 then
            	   tmp[ii] = t2
                else
            	   tmp[ii] = t1
                end
            end
        end
        skel3D[idx2] = skel3D[idx1]:clone() + tmp:clone() / torch.norm(tmp) * avglen[id]
    end
        
    skel3D = skel3D:double()
    local outputImage = '../output/dslr_dance1/' .. string.format('%04d.png', idx)
    image.save(outputImage, inp)
    local outputFile = '../output/dslr_dance1/' .. string.format('%04d.json', idx)
    local t = {}
    for i=1,skel3D:size(1) do
        t[i] = {}
        for j=1,skel3D:size(2) do
            t[i][j] = skel3D[i][j]
        end
    end
    local str = json.encode(t)
    local f = io.open(outputFile, 'w')
    f:write(str)
    f:close()
end


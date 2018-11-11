require "json"

function loadInTheWild(seqName, idx)
	local imgName = seqName .. string.format('_%012d_', idx - 1) .. 'rendered.png'
	local fullImgName = '/media/posefs1b/Users/donglaix/siggasia018/' .. seqName .. '/openpose_image/' .. imgName
	local img = image.load(fullImgName)

	local jsonName = seqName .. string.format('_%012d_', idx - 1) .. 'keypoints.json'
	local fullJsonName = '/media/posefs1b/Users/donglaix/siggasia018/' .. seqName .. '/openpose_result/' .. jsonName
	local f = io.open(fullJsonName)
	local content = f:read "*a"
	local data = json.decode(content)

	local high_score = 0.0
	local high_ip = 0
	for ip = 1, #data['people'] do
		local joints = torch.Tensor(data['people'][ip]['pose_keypoints_2d'])
		local reshaped = torch.reshape(joints, 25, 3)
		local sum_score = torch.sum(reshaped[{{}, 3}])
		if sum_score > high_score then
			high_score = sum_score
			high_ip = ip
		end
	end

	local joints = torch.Tensor(data['people'][high_ip]['pose_keypoints_2d'])
	local score = torch.reshape(joints, 25, 3)[{{}, 3}]
	local bool = torch.ge(score, 0.2)
	local reshaped = torch.reshape(joints, 25, 3)[{{}, {1,2}}]
	local X = reshaped[{{}, 1}]
	local Y = reshaped[{{}, 2}]
	X = X[bool]
	Y = Y[bool]
	local jmax = torch.Tensor({torch.max(X), torch.max(Y)})
	local jmin = torch.Tensor({torch.min(X), torch.min(Y)})
	local center = (jmax + jmin) / 2
	local height = jmax[2] - jmin[2]
	local s = height * 1.5 / 200
	local inp = crop(img, center, s, 0, opt.inputRes)

	return inp
end
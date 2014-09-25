function showMovie(str,num)

	image = imread(strcat(str,'.jpg'));
	partSize = 192;
	rows = floor(size(image,1)/partSize);
	cols = floor(size(image,2)/partSize);
	numOfParts = rows*cols;
	pcs = cutImageToParts;

	load(strcat(str,'_',num2str(num)));
	for i = 1:size(frames,1)
		if frames(i,1) == 0, break; end
		disp(frames(i,:));
		pause(0.1);
	end

	function pcs = cutImageToParts()
		for index = 1 : numOfParts
		  rowStartIndex = (ceil(index / cols)  - 1) * partSize + 1;
		  rowEndIndex = rowStartIndex + (partSize -  1);
		  colStartIndex = mod(index - 1, cols)  * partSize + 1;
		  colEndIndex = colStartIndex + (partSize -  1);
		  pcs{index} = image(rowStartIndex :rowEndIndex, colStartIndex :colEndIndex, :);
		end
	end

	function disp(order)
	    fg = [];
	    for i = 1:rows
	      fg = cat(1,fg,cat(2,pcs{order((1:cols)+(i-1)*cols)}));
	    end
	    imshow(fg);
	    %imwrite(fg,'2.png','png');
	end

end


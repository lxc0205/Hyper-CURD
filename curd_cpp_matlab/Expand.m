function expand = Expand
    expand.Expand_cross = @Expand_cross;
    expand.Expand_base = @Expand_base;    
end

function Mssim_expand = Expand_cross(Mssim)
    Mssim0 = [Mssim Mssim.^2 Mssim.^(1/2) Mssim.^3 Mssim.^(1/3) log(Mssim) 2.^Mssim exp(Mssim)];
    Mssim_add = [];
    for x_i = 1:size(Mssim0,2)
        for y_i = (x_i+1):size(Mssim0,2)
            Mssim_add = [Mssim_add, Mssim0(:,x_i).*Mssim0(:,y_i)];
        end
    end
    Mssim_expand =[Mssim0 Mssim_add];
end

function Mssim_expand = Expand_base(Mssim)
    Mssim_expand = real([Mssim Mssim.^2 Mssim.^(1/2) Mssim.^3 Mssim.^(1/3) log(Mssim) 2.^Mssim exp(Mssim)]);
end
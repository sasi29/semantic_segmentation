classdef paddingLayer < nnet.layer.Layer
    % A padding layer. Pads input feature maps with zeros.
    %
    % layer = paddingLayer(padSize) pads first two dimensions of a feature
    % map with padSize zeros.
    
    properties
        PadSize
    end
    methods
        function this = paddingLayer(padSize)
            assert(isscalar(padSize),'padSize must be a scalar');
            this.PadSize = repelem(padSize,1,2);
        end
        
        function Z = predict(this, X) 
            % add padding.
            Z = padarray(X,this.PadSize);
        end
        
        function dLdX = backward(this,~,~,dLdZ,~)
            % remove padding.
            [H,W,~] = size(dLdZ);
            r1 = this.PadSize(1)+1;
            r2 = H - this.PadSize(1);
            c1 = this.PadSize(2)+1;
            c2 = W - this.PadSize(2);
            dLdX = dLdZ(r1:r2,c1:c2,:,:);
        end
    end
    
    
end
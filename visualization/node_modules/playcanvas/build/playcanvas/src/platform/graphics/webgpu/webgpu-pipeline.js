class WebgpuPipeline {
		constructor(device){
				this.device = device;
		}
		getPipelineLayout(bindGroupFormats) {
				const bindGroupLayouts = [];
				bindGroupFormats.forEach((format)=>{
						bindGroupLayouts.push(format.bindGroupLayout);
				});
				const desc = {
						bindGroupLayouts: bindGroupLayouts
				};
				const pipelineLayout = this.device.wgpu.createPipelineLayout(desc);
				return pipelineLayout;
		}
}

export { WebgpuPipeline };

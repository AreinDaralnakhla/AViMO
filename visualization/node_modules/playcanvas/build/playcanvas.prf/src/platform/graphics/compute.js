class ComputeParameter {
		constructor(){
				this.scopeId = null;
		}
}
class Compute {
		constructor(graphicsDevice, shader, name = 'Unnamed'){
				this.shader = null;
				this.parameters = new Map();
				this.countX = 1;
				this.device = graphicsDevice;
				this.shader = shader;
				this.name = name;
				if (graphicsDevice.supportsCompute) {
						this.impl = graphicsDevice.createComputeImpl(this);
				}
		}
		setParameter(name, value) {
				let param = this.parameters.get(name);
				if (!param) {
						param = new ComputeParameter();
						param.scopeId = this.device.scope.resolve(name);
						this.parameters.set(name, param);
				}
				param.value = value;
		}
		getParameter(name) {
				return this.parameters.get(name)?.value;
		}
		deleteParameter(name) {
				this.parameters.delete(name);
		}
		applyParameters() {
				for (const [, param] of this.parameters){
						param.scopeId.setValue(param.value);
				}
		}
		setupDispatch(x, y, z) {
				this.countX = x;
				this.countY = y;
				this.countZ = z;
		}
}

export { Compute };

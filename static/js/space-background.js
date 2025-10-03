// Advanced Space Background with Particles and Nebulae
class SpaceBackground {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.stars = [];
        this.particles = [];
        this.nebulaClouds = [];
        
        document.getElementById('universe').appendChild(this.canvas);
        this.init();
    }

    init() {
        this.resize();
        this.createStars();
        this.createParticles();
        this.createNebulaClouds();
        this.animate();
        
        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.createStars(); // Recreate stars on resize
    }

    createStars() {
        this.stars = [];
        const starCount = 400;
        
        for (let i = 0; i < starCount; i++) {
            this.stars.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                size: Math.random() * 2 + 0.5,
                speed: Math.random() * 0.3 + 0.1,
                brightness: Math.random() * 0.8 + 0.2,
                twinkleSpeed: Math.random() * 0.05 + 0.02,
                twinkleOffset: Math.random() * Math.PI * 2
            });
        }
    }

    createParticles() {
        this.particles = [];
        const particleCount = 50;
        
        for (let i = 0; i < particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.2,
                vy: (Math.random() - 0.5) * 0.2,
                size: Math.random() * 1 + 0.5,
                life: 1,
                decay: Math.random() * 0.002 + 0.001
            });
        }
    }

    createNebulaClouds() {
        this.nebulaClouds = [];
        const cloudCount = 3;
        
        for (let i = 0; i < cloudCount; i++) {
            this.nebulaClouds.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                radius: Math.random() * 200 + 100,
                color: i === 0 ? 'rgba(79, 195, 247, 0.1)' : 
                       i === 1 ? 'rgba(186, 104, 200, 0.08)' : 
                       'rgba(77, 182, 172, 0.06)',
                pulseSpeed: Math.random() * 0.002 + 0.001,
                pulseOffset: Math.random() * Math.PI * 2
            });
        }
    }

    animate() {
        this.ctx.fillStyle = 'rgba(10, 14, 23, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.drawNebulaClouds();
        this.drawStars();
        this.drawParticles();

        requestAnimationFrame(() => this.animate());
    }

    drawNebulaClouds() {
        const time = Date.now() * 0.001;
        
        this.nebulaClouds.forEach(cloud => {
            const pulse = Math.sin(time * cloud.pulseSpeed + cloud.pulseOffset) * 0.3 + 0.7;
            const radius = cloud.radius * pulse;
            
            const gradient = this.ctx.createRadialGradient(
                cloud.x, cloud.y, 0,
                cloud.x, cloud.y, radius
            );
            
            gradient.addColorStop(0, cloud.color);
            gradient.addColorStop(1, 'transparent');
            
            this.ctx.globalCompositeOperation = 'lighter';
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(cloud.x, cloud.y, radius, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.globalCompositeOperation = 'source-over';
        });
    }

    drawStars() {
        const time = Date.now() * 0.001;
        
        this.stars.forEach(star => {
            // Twinkling effect
            const twinkle = Math.sin(time * star.twinkleSpeed + star.twinkleOffset) * 0.3 + 0.7;
            const brightness = star.brightness * twinkle;
            
            this.ctx.beginPath();
            this.ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(255, 255, 255, ${brightness})`;
            this.ctx.fill();

            // Star movement
            star.y += star.speed;
            if (star.y > this.canvas.height) {
                star.y = 0;
                star.x = Math.random() * this.canvas.width;
            }
        });
    }

    drawParticles() {
        this.particles.forEach(particle => {
            particle.x += particle.vx;
            particle.y += particle.vy;
            particle.life -= particle.decay;

            // Wrap around edges
            if (particle.x < 0) particle.x = this.canvas.width;
            if (particle.x > this.canvas.width) particle.x = 0;
            if (particle.y < 0) particle.y = this.canvas.height;
            if (particle.y > this.canvas.height) particle.y = 0;

            // Respawn dead particles
            if (particle.life <= 0) {
                particle.x = Math.random() * this.canvas.width;
                particle.y = Math.random() * this.canvas.height;
                particle.life = 1;
            }

            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(79, 195, 247, ${particle.life * 0.3})`;
            this.ctx.fill();
        });
    }
}

// Initialize space background
document.addEventListener('DOMContentLoaded', () => {
    new SpaceBackground();
});
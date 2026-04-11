import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mem4ristor.arena import Arena

def demo_arena_evolution():
    print("\n⚔️ DEMO: The Arena (Adversarial Co-Evolution)")
    print("="*60)
    
    # 1. Initialize Arena
    arena = Arena(seed=42)
    print("1. Gates Open. Predator vs Prey.")
    
    # 2. Run Combat (1000 Rounds)
    rounds = 1000
    history_win = []
    history_error = []
    
    print(f"2. Fighting {rounds} rounds...")
    
    for r in range(rounds):
        result = arena.fight_round()
        
        # Log 1 for Predator Win, 0 for Prey Win
        history_win.append(1 if result['winner'] == 'Predator' else 0)
        history_error.append(result['error'])
        
        if r % 100 == 0:
            print(f"   Round {r}: Winner={result['winner']}, Error={result['error']:.4f}")
            
    # 3. Analyze Evolution
    # Moving average of win rate (Window 50)
    window = 50
    win_rate = np.convolve(history_win, np.ones(window)/window, mode='valid')
    
    # 4. Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(win_rate, label='Predator Win Rate')
    plt.axhline(0.5, color='r', linestyle='--', label='Equilibrium')
    plt.title("Adversarial Co-Evolution: Arms Race")
    plt.xlabel("Rounds")
    plt.ylabel("Predator Dominance")
    plt.legend()
    plt.grid(True)
    
    output_path = os.path.join(os.path.dirname(__file__), 'arena_result.png')
    plt.savefig(output_path)
    print(f"\n✅ Result saved to: {output_path}")
    
    # 5. Verdict
    avg_win = np.mean(history_win)
    print(f"\n5. Verdict:")
    print(f"   Global Win Rate (Predator): {avg_win:.2f}")
    
    if 0.3 < avg_win < 0.7:
        print(f"✅ SUCCESS: Balanced Evolution.")
        print(f"   Neither side dominates. The arms race is active.")
    else:
        print(f"❌ FAILURE: Dominance detected ({avg_win:.2f}).")
        if avg_win > 0.7:
            print("   Predator is too strong.")
        else:
            print("   Prey is too elusive.")

if __name__ == "__main__":
    demo_arena_evolution()

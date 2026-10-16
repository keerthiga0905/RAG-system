import React, { useState } from 'react';
import { 
  User, Mail, Lock, ShieldCheck, 
  GraduationCap, Building2, Compass, ArrowRight, CheckCircle2 
} from 'lucide-react';
import type { UserProfile } from '../types';

interface LoginPageProps {
  onLogin: (profile: UserProfile) => void;
  activeUser?: UserProfile | null;
  onCancel?: () => void;
}

const PRESET_ACCOUNTS: UserProfile[] = [
  {
    email: 'keerthiga@cit.edu.in',
    name: 'Keerthiga M',
    role: 'Student',
    department: 'M.Sc. Software Systems',
    avatarColor: '#800000',
    createdAt: '2026-10-15'
  },
  {
    email: 'manjula.gandhi@cit.edu.in',
    name: 'Dr. S. Manjula Gandhi',
    role: 'Faculty',
    department: 'Associate Professor & HOD (I/c) - M.Sc. SS',
    avatarColor: '#1D4ED8',
    createdAt: '2026-10-14'
  },
  {
    email: 'admissions.candidate@cit.edu.in',
    name: 'Prospective Aspirant',
    role: 'Aspirant',
    department: 'UG Admissions (B.E. / B.Tech)',
    avatarColor: '#059669',
    createdAt: '2026-10-16'
  },
  {
    email: 'hostel.resident@cit.edu.in',
    name: 'Hostel Resident Scholar',
    role: 'Resident',
    department: 'CIT Campus Hostels',
    avatarColor: '#7C3AED',
    createdAt: '2026-10-16'
  }
];

export const LoginPage: React.FC<LoginPageProps> = ({ onLogin, activeUser, onCancel }) => {
  const [email, setEmail] = useState<string>('');
  const [name, setName] = useState<string>('');
  const [role, setRole] = useState<string>('Student');
  const [department, setDepartment] = useState<string>('M.Sc. Software Systems');
  const [password, setPassword] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string>('');

  const handleCustomLogin = (e: React.FormEvent) => {
    e.preventDefault();
    const cleanEmail = email.trim().toLowerCase();
    if (!cleanEmail || !cleanEmail.includes('@')) {
      setErrorMsg('Please enter a valid institutional or personal email address.');
      return;
    }

    const displayName = name.trim() || cleanEmail.split('@')[0].replace('.', ' ').toUpperCase();
    const profile: UserProfile = {
      email: cleanEmail,
      name: displayName,
      role: role || 'Student',
      department: department.trim() || 'General CIT',
      avatarColor: cleanEmail.includes('cit.edu.in') ? '#800000' : '#2563EB',
      createdAt: new Date().toISOString()
    };

    setErrorMsg('');
    onLogin(profile);
  };

  const handleSelectPreset = (preset: UserProfile) => {
    onLogin(preset);
  };

  return (
    <div className="login-page-container">
      <div className="login-card-wrapper">
        
        {/* Left Branding Panel */}
        <div className="login-brand-panel">
          <div className="brand-logo-badge">
            <img src="/cit_logo.svg" alt="CIT Logo" className="login-brand-logo" />
            <div>
              <h2 className="login-brand-heading">Coimbatore Institute of Technology</h2>
              <span className="login-brand-sub">Autonomous Institution Grounded Knowledge AI</span>
            </div>
          </div>

          <div className="login-benefits">
            <div className="benefit-item">
              <ShieldCheck className="benefit-icon" size={20} />
              <div>
                <strong>Isolated History per Email</strong>
                <p>Every account maintains its own verified inquiries, roadmap history, and search records.</p>
              </div>
            </div>

            <div className="benefit-item">
              <GraduationCap className="benefit-icon" size={20} />
              <div>
                <strong>3-Level Hierarchical Grounding</strong>
                <p>Level 1 Dataset, Level 2 Official Website (cit.edu.in), and Level 3 Campus Regulations.</p>
              </div>
            </div>

            <div className="benefit-item">
              <Compass className="benefit-icon" size={20} />
              <div>
                <strong>Campus Roadmap Navigation</strong>
                <p>Interactive turn-by-turn walking directions across CIT landmarks and academic blocks.</p>
              </div>
            </div>
          </div>

          <div className="login-brand-footer">
            <span>Machine Learning Laboratory Project &bull; Coimbatore, Tamil Nadu</span>
          </div>
        </div>

        {/* Right Form Panel */}
        <div className="login-form-panel">
          <div className="login-form-header">
            <h3>{activeUser ? 'Switch Account' : 'Sign In to Your Workspace'}</h3>
            <p>Enter your institutional or personal email address to load your personal chat history and dashboard.</p>
          </div>

          {errorMsg && (
            <div className="login-error-banner">
              {errorMsg}
            </div>
          )}

          {/* Quick 1-Click Demo Profiles */}
          <div className="preset-accounts-section">
            <span className="section-label">Quick Sign-In Profiles</span>
            <div className="preset-grid">
              {PRESET_ACCOUNTS.map((preset) => (
                <button
                  key={preset.email}
                  type="button"
                  className={`preset-card ${activeUser?.email === preset.email ? 'active' : ''}`}
                  onClick={() => handleSelectPreset(preset)}
                >
                  <div 
                    className="preset-avatar" 
                    style={{ backgroundColor: preset.avatarColor }}
                  >
                    {preset.name.charAt(0)}
                  </div>
                  <div className="preset-info">
                    <div className="preset-name">{preset.name}</div>
                    <div className="preset-email">{preset.email}</div>
                    <div className="preset-role">{preset.department}</div>
                  </div>
                  {activeUser?.email === preset.email && (
                    <CheckCircle2 size={16} className="preset-check" />
                  )}
                </button>
              ))}
            </div>
          </div>

          <div className="login-divider">
            <span>OR ENTER CUSTOM EMAIL</span>
          </div>

          {/* Custom Email Form */}
          <form onSubmit={handleCustomLogin} className="login-form">
            <div className="form-group">
              <label htmlFor="user-email">
                <Mail size={14} /> Email Address
              </label>
              <input
                id="user-email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="e.g. student@cit.edu.in or personal@email.com"
                required
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label htmlFor="user-name">
                  <User size={14} /> Full Name (Optional)
                </label>
                <input
                  id="user-name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Keerthiga M"
                />
              </div>

              <div className="form-group">
                <label htmlFor="user-role">
                  <GraduationCap size={14} /> Role
                </label>
                <select
                  id="user-role"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                >
                  <option value="Student">Student</option>
                  <option value="Faculty">Faculty / Professor</option>
                  <option value="Research Scholar">Research Scholar</option>
                  <option value="Parent/Visitor">Parent / Visitor</option>
                  <option value="Alumni">Alumni</option>
                </select>
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="user-dept">
                <Building2 size={14} /> Department / Programme
              </label>
              <input
                id="user-dept"
                type="text"
                value={department}
                onChange={(e) => setDepartment(e.target.value)}
                placeholder="e.g. M.Sc. Software Systems"
              />
            </div>

            <div className="form-group">
              <label htmlFor="user-pwd">
                <Lock size={14} /> Password / PIN (Optional)
              </label>
              <input
                id="user-pwd"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="•••••••• (optional for demo access)"
              />
            </div>

            <div className="form-actions">
              {onCancel && (
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={onCancel}
                >
                  Cancel
                </button>
              )}
              <button
                type="submit"
                className="btn-primary login-submit-btn"
                id="login-submit-btn"
              >
                <span>Continue to CIT AI</span>
                <ArrowRight size={16} />
              </button>
            </div>
          </form>

        </div>
      </div>
    </div>
  );
};

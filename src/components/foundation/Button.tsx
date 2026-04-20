import React from 'react';
import { tokens } from '../../tokens/tokens';
import './Button.css';

export interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  state?: 'idle' | 'loading' | 'disabled' | 'success' | 'error';
  className?: string;
  children: React.ReactNode;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
  disabled?: boolean;
  title?: string;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  state = 'idle',
  className = '',
  children,
  onClick,
  type = 'button',
  disabled = false,
  title,
}) => {
  const isLoading = state === 'loading';
  const isDisabled = state === 'disabled' || disabled || isLoading;

  return (
    <button
      type={type}
      className={`btn btn-${variant} btn-${size} ${state === 'loading' ? 'is-loading' : ''} ${
        isDisabled ? 'is-disabled' : ''
      } ${className}`}
      onClick={onClick}
      disabled={isDisabled}
      title={title}
      style={{
        fontFamily: tokens.typography.fontFamily.body,
        fontSize: size === 'sm' ? tokens.typography.fontSize.sm : 
                  size === 'lg' ? tokens.typography.fontSize.lg :
                  tokens.typography.fontSize.base,
      }}
    >
      {isLoading && <span className="btn-loader" />}
      <span className="btn-content">{children}</span>
    </button>
  );
};
